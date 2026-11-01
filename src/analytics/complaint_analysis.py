
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, round as spark_round,
    date_format, desc, when
)

# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "nyc_311_cleaned"
OUTPUT_DIR = BASE_DIR / "reports" / "analytics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start Spark
spark = (
    SparkSession.builder
    .appName("SmartCity311_ComplaintAnalytics")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    print("\nLoading processed complaint dataset...")

    df = spark.read.parquet(str(INPUT_PATH))

    print("Total complaints:", df.count())

    # 1. Top 10 complaint types
    print("\n1. TOP 10 COMPLAINT TYPES")

    top_complaints = (
        df.groupBy("complaint_type")
        .agg(count("*").alias("total_complaints"))
        .orderBy(desc("total_complaints"))
        .limit(10)
    )

    top_complaints.show(truncate=False)

    top_complaints.toPandas().to_csv(
        OUTPUT_DIR / "top_complaints.csv",
        index=False
    )

    # 2. Open vs Closed complaints
    print("\n2. RESOLUTION STATUS")

    status_analysis = (
        df.groupBy("resolution_status")
        .agg(count("*").alias("total_complaints"))
        .orderBy(desc("total_complaints"))
    )

    status_analysis.show()

    status_analysis.toPandas().to_csv(
        OUTPUT_DIR / "resolution_status.csv",
        index=False
    )

    # 3. Average resolution time
    print("\n3. AVERAGE RESOLUTION TIME")

    resolution_analysis = (
        df.filter(
            (col("resolution_status") == "Closed") &
            col("resolution_hours").isNotNull() &
            (col("resolution_hours") >= 0)
        )
        .groupBy("complaint_type")
        .agg(
            spark_round(
                avg("resolution_hours"), 2
            ).alias("average_resolution_hours"),
            count("*").alias("closed_complaints")
        )
        .orderBy(desc("average_resolution_hours"))
    )

    resolution_analysis.show(20, truncate=False)

    resolution_analysis.toPandas().to_csv(
        OUTPUT_DIR / "resolution_time_by_type.csv",
        index=False
    )

    # 4. Monthly complaint trends
    print("\n4. MONTHLY COMPLAINT TRENDS")

    monthly_analysis = (
    df.groupBy("complaint_year", "complaint_month")
    .agg(count("*").alias("total_complaints"))
    .orderBy("complaint_year", "complaint_month")
)

    monthly_analysis.show(50)

    monthly_analysis.toPandas().to_csv(
        OUTPUT_DIR / "monthly_trends.csv",
        index=False
    )

    # 5. Borough-wise complaints
    print("\n5. BOROUGH-WISE COMPLAINTS")

    borough_analysis = (
        df.groupBy("borough")
        .agg(count("*").alias("total_complaints"))
        .orderBy(desc("total_complaints"))
    )

    borough_analysis.show(truncate=False)

    borough_analysis.toPandas().to_csv(
        OUTPUT_DIR / "borough_analysis.csv",
        index=False
    )

    # 6. Department-wise complaints
    print("\n6. DEPARTMENT-WISE COMPLAINTS")

    department_analysis = (
        df.groupBy("agency_name")
        .agg(count("*").alias("total_complaints"))
        .orderBy(desc("total_complaints"))
    )

    department_analysis.show(20, truncate=False)

    department_analysis.toPandas().to_csv(
        OUTPUT_DIR / "department_analysis.csv",
        index=False
    )

    print("\nCOMPLAINT ANALYTICS COMPLETED!")
    print("Reports saved in:", OUTPUT_DIR)

finally:
    spark.stop()
