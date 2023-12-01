import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.sql.functions import sum as sql_sum
from pyspark.sql.functions import udf, when
from pyspark.sql.types import IntegerType

# Initialize Spark Context and Session
spark = SparkSession.builder.appName("DecisionTreeModel").getOrCreate()

features = ["credit_age", "income", "full_time", "part_time", "student"]
label = "credit_approved"

# Model metadata extraction
model_df = spark.read.parquet("./src/model/dtc_model/data/*")

# Graph creation
node_cols = [
    "id",
    "prediction",
    "impurity",
    "impurityStats",
    "rawCount",
    "gain",
    "leftChild",
    "rightChild",
    "split",
]

# Convert metatdata to list
noderows = model_df.select(*node_cols).collect()

# Creates Nodes
G = nx.Graph()
for rw in noderows:
    if rw["leftChild"] < 0 and rw["rightChild"] < 0:
        G.add_node(rw["id"], cat="Prediction", predval=rw["prediction"])
    else:
        G.add_node(
            rw["id"],
            cat="splitter",
            featureIndex=rw["split"]["featureIndex"],
            thresh=rw["split"]["leftCategoriesOrThreshold"],
            leftChild=rw["leftChild"],
            rightChild=rw["rightChild"],
            numCat=rw["split"]["numCategories"],
        )

# Adding edges to the graph
for rw in model_df.filter("leftChild > 0 and rightChild > 0").collect():
    tempnode = G.nodes()[rw["id"]]
    G.add_edge(
        rw["id"],
        rw["leftChild"],
        reason="{0} < {1}".format(
            features[tempnode["featureIndex"]], tempnode["thresh"][0]
        ),
    )
    G.add_edge(
        rw["id"],
        rw["rightChild"],
        reason="{0} > {1}".format(
            features[tempnode["featureIndex"]], tempnode["thresh"][0]
        ),
    )

# RDD creation and transformation
rdd_noderows = spark.sparkContext.parallelize(noderows).persist()

# Create dictionary to save decision paths
index_to_path_dct = {}
for n in G.nodes():
    p = nx.shortest_path(G, 0, n)
    index_to_path_dct[str(n)] = " AND ".join(
        G.get_edge_data(p[i], p[i + 1])["reason"] for i in range(len(p) - 1)
    )

# UDF for label count extraction
extract_label = udf(lambda v: int(v[1]), IntegerType())

df_sql_rules = (
    rdd_noderows.toDF()
    .select(
        "id",
        col("id").cast("string").alias("sql_rules"), # Duplicating id as string
        extract_label("impurityStats").alias(label),
        col("rawCount").alias("total"),
    )
    .replace( 
        to_replace=index_to_path_dct, #  Map duplicate ID to add sql rules
        subset=["sql_rules"]
    )
)

df_sql_rules.show(20, False)

csv_path = "./src/sample_data/data.csv"
training_data = spark.read.csv(csv_path, header=True, inferSchema=True)
training_data.createOrReplaceTempView("temp_data")

query = """
SELECT *
FROM temp_data
WHERE income > 49000.0 AND full_time > 0.5 AND credit_age > 39.5 AND credit_age > 43.5
"""

filtered_df = spark.sql(query)
filtered_df.show()
