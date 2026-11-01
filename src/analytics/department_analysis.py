
from pathlib import Path

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    median,
    expr,
    sum as spark_sum,
    when,
    desc
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "nyc_311_cleaned"

REPORT_DIR = BASE_DIR / "reports" / "department"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. START SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("SmartCity311_DepartmentAnalytics")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:

    print("\nLoading department complaint data...")

    df = spark.read.parquet(str(INPUT_PATH))

    print("Total records:", df.count())


    # ==========================================
    # 3. DEPARTMENT WORKLOAD
    # ==========================================

    print("\n1. DEPARTMENT-WISE COMPLAINT WORKLOAD")

    department_workload = (
        df.groupBy("agency_name")
        .agg(
            count("*").alias("total_complaints"),

            spark_sum(
                when(col("resolution_status") == "Closed", 1)
                .otherwise(0)
            ).alias("closed_complaints"),

            spark_sum(
                when(col("resolution_status") == "Open", 1)
                .otherwise(0)
            ).alias("open_complaints")
        )
        .withColumn(
            "closure_percentage",
            expr(
                "ROUND(closed_complaints * 100.0 / "
                "total_complaints, 2)"
            )
        )
        .orderBy(desc("total_complaints"))
    )

    department_workload.show(30, truncate=False)

    department_workload.toPandas().to_csv(
        REPORT_DIR / "department_workload.csv",
        index=False
    )


    # ==========================================
    # 4. RESOLUTION PERFORMANCE
    # ==========================================

    print("\n2. DEPARTMENT RESOLUTION PERFORMANCE")

    closed_df = df.filter(
        (col("resolution_status") == "Closed") &
        col("resolution_hours").isNotNull() &
        (col("resolution_hours") >= 0)
    )

    resolution_performance = (
        closed_df.groupBy("agency_name")
        .agg(
            count("*").alias("closed_sample_count"),

            avg("resolution_hours").alias(
                "average_resolution_hours"
            ),

            median("resolution_hours").alias(
                "median_resolution_hours"
            ),

            expr(
                "percentile_approx(resolution_hours, 0.90)"
            ).alias("p90_resolution_hours")
        )
        .filter(col("closed_sample_count") >= 30)
        .orderBy(desc("average_resolution_hours"))
    )

    resolution_performance.show(30, truncate=False)

    resolution_performance.toPandas().to_csv(
        REPORT_DIR / "department_resolution_performance.csv",
        index=False
    )


    # ==========================================
    # 5. PENDING COMPLAINT ANALYSIS
    # ==========================================

    print("\n3. DEPARTMENT PENDING COMPLAINT ANALYSIS")

    pending_analysis = (
        df.filter(col("resolution_status") == "Open")
        .groupBy("agency_name")
        .agg(
            count("*").alias("pending_complaints")
        )
        .orderBy(desc("pending_complaints"))
    )

    pending_analysis.show(30, truncate=False)

    pending_analysis.toPandas().to_csv(
        REPORT_DIR / "department_pending_analysis.csv",
        index=False
    )


    # ==========================================
    # 6. FINAL OUTPUT
    # ==========================================

    print("\n" + "=" * 50)
    print("DEPARTMENT ANALYTICS COMPLETED!")
    print("=" * 50)

    print("Reports saved in:", REPORT_DIR)

finally:

    spark.stop()
