import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# Initialize Spark Context and Session
spark = SparkSession.builder.appName("SampleData").getOrCreate()

data = [
    (25, 50000, "Full-time", 1),
    (42, 80000, "Part-time", 0),
    (35, 60000, "Full-time", 1),
    (29, 45000, "Unemployed", 0),
    (48, 90000, "Full-time", 1),
    (33, 55000, "Full-time", 1),
    (40, 70000, "Part-time", 0),
    (22, 35000, "Student", 0),
    (45, 85000, "Full-time", 1),
    (30, 40000, "Unemployed", 0),
    (31, 65000, "Full-time", 1),
    (28, 48000, "Part-time", 0),
    (50, 95000, "Full-time", 1),
    (37, 52000, "Unemployed", 0),
    (41, 68000, "Part-time", 1),
    (26, 56000, "Student", 0),
    (39, 72000, "Full-time", 1),
    (34, 47000, "Part-time", 0),
    (47, 88000, "Unemployed", 1),
    (32, 41000, "Student", 0),
    (27, 53000, "Full-time", 1),
    (44, 77000, "Part-time", 0),
    (36, 62000, "Unemployed", 1),
    (49, 91000, "Full-time", 0),
    (38, 57000, "Student", 1),
    (43, 81000, "Full-time", 0),
    (46, 84000, "Part-time", 1),
    (21, 34000, "Unemployed", 0),
    (23, 36000, "Student", 1),
    (24, 39000, "Full-time", 0),
]

# Load data and cache it
# data = spark.table("your_data").cache()

schema = StructType(
    [
        StructField("credit_age", IntegerType(), True),
        StructField("income", IntegerType(), True),
        StructField("employment_status", StringType(), True),
        StructField("credit_approved", IntegerType(), True),
    ]
)

ohe = (
    when(col("employment_status") == v, 1).alias(k)
    for k, v in {
        "full_time": "Full-time",
        "part_time": "Part-time",
        # "student": "Student",
    }.items()
)

# Create DataFrame
df = spark.createDataFrame(data, schema).select(
    "credit_age",
    "income",
    *ohe,
    "credit_approved",
)

df.show()

csv_path = "./src/sample_data/data.csv"

df.write.mode("overwrite").csv(csv_path, header=True)
