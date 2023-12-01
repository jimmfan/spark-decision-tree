import networkx as nx
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.sql.functions import sum as sql_sum
from pyspark.sql.functions import udf, when
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Initialize Spark Context and Session
spark = SparkSession.builder.appName("DecisionTreeModel").getOrCreate()

# Define features
features = ["credit_age", "income", "full_time", "part_time", "student"]
label = "credit_approved"

# Decision Tree Classifier
params = {
    "featuresCol": "features",
    "labelCol": label,
    "predictionCol": "prediction",
    "probabilityCol": "probability",
    "rawPredictionCol": "rawPrediction",
    "maxDepth": 8,
    "maxBins": 32,
    "minInstancesPerNode": 1,
    "minInfoGain": 0.0,
    "maxMemoryInMB": 256,
    "cacheNodeIds": False,
    "checkpointInterval": 10,
    "impurity": "gini",
    "seed": 42,
}

csv_path = "./src/sample_data/data.csv"
df = spark.read.csv(csv_path, header=True, inferSchema=True)

# VectorAssembler to create feature vector for spark
assembler = VectorAssembler(inputCols=features, outputCol=params["featuresCol"])

# Transform the data to
final_data = assembler.transform(df.fillna(0))

# Decision Tree Classifier
dtc = DecisionTreeClassifier(**params)

# Fit the model
model = dtc.fit(final_data)

# Make predictions
predictions = model.transform(final_data)

# Print Features
print(model.toDebugString)

# Save the model
model.write().overwrite().save("./src/model/dtc_model")
