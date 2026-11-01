from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, when, lit,
    datediff, hour, dayofweek, month,
    year, dayofmonth, round as spark_round
)
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# 1. Start Spark
spark = SparkSession.builder \
    .appName("SmartCity311_DataProcessing") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

INPUT = "data/raw/nyc_311_sample.csv"
OUTPUT = "data/processed/nyc_311_cleaned"

print("\nLoading raw dataset...")

df = spark.read.option("header", True) \
    .option("inferSchema", True) \
    .csv(INPUT)

print("Original records:", df.count())

# 2. Remove duplicate complaint IDs
df = df.dropDuplicates(["unique_key"])

print("After duplicate removal:", df.count())

# 3. Handle missing values
df = df.fillna({
    "descriptor": "Unknown",
    "incident_zip": "Unknown"
})

# 4. Convert timestamps
df = df.withColumn(
    "created_date",
    to_timestamp(col("created_date"))
).withColumn(
    "closed_date",
    to_timestamp(col("closed_date"))
)

# 5. Calculate resolution time in hours
df = df.withColumn(
    "resolution_hours",
    when(
        col("closed_date").isNotNull() &
        (col("closed_date") >= col("created_date")),
        (col("closed_date").cast("long") -
         col("created_date").cast("long")) / 3600
    ).otherwise(None)
)

# 6. Create resolution status
df = df.withColumn(
    "resolution_status",
    when(col("closed_date").isNull(), "Open")
    .otherwise("Closed")
)

# 7. Extract time-based features
df = df.withColumn(
    "complaint_year", year("created_date")
).withColumn(
    "complaint_month", month("created_date")
).withColumn(
    "complaint_day", dayofmonth("created_date")
).withColumn(
    "complaint_hour", hour("created_date")
).withColumn(
    "complaint_weekday", dayofweek("created_date")
)

# 8. Create complaint priority categories
df = df.withColumn(
    "priority",
    when(
        col("complaint_type").isin(
            "HEAT/HOT WATER",
            "WATER LEAK",
            "PLUMBING"
        ), "High"
    ).when(
        col("complaint_type").isin(
            "Street Light Condition",
            "Blocked Driveway"
        ), "Medium"
    ).otherwise("Normal")
)

# 9. Keep location information as missing
# Do not invent latitude/longitude values.

# 10. Save processed data in Parquet format
df.write.mode("overwrite").parquet(OUTPUT)

print("\nDATA PROCESSING COMPLETED!")
print("Final records:", df.count())
print("Total columns:", len(df.columns))

print("\nResolution status:")
df.groupBy("resolution_status").count().show()

print("\nPriority distribution:")
df.groupBy("priority").count().show()

print("\nProcessed dataset preview:")
df.select(
    "unique_key",
    "complaint_type",
    "resolution_status",
    "resolution_hours",
    "priority"
).show(10, truncate=False)

print("\nSaved to:", OUTPUT)

spark.stop()
