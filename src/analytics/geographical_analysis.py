
from pathlib import Path

import pandas as pd
import folium

from folium.plugins import HeatMap, FastMarkerCluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc
from sklearn.cluster import KMeans


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "nyc_311_cleaned"

REPORT_DIR = BASE_DIR / "reports" / "geographical"
MAP_DIR = BASE_DIR / "reports" / "maps"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. START SPARK
# ==========================================

spark = (
    SparkSession.builder
    .appName("SmartCity311_GeographicalAnalytics")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:

    print("\nLoading geographical complaint data...")

    df = spark.read.parquet(str(INPUT_PATH))

    print("Total records:", df.count())


    # ==========================================
    # 3. BOROUGH-WISE ANALYSIS
    # ==========================================

    print("\n1. BOROUGH-WISE COMPLAINT DISTRIBUTION")

    borough_analysis = (
        df.groupBy("borough")
        .agg(count("*").alias("total_complaints"))
        .orderBy(desc("total_complaints"))
    )

    borough_analysis.show(truncate=False)

    borough_analysis.toPandas().to_csv(
        REPORT_DIR / "borough_geographical_analysis.csv",
        index=False
    )


    # ==========================================
    # 4. VALIDATE COORDINATES
    # ==========================================

    print("\n2. VALIDATING LOCATION COORDINATES")

    geo_df = (
        df.select(
            "unique_key",
            "complaint_type",
            "borough",
            col("latitude").cast("double").alias("latitude"),
            col("longitude").cast("double").alias("longitude")
        )
        .filter(
            col("latitude").isNotNull() &
            col("longitude").isNotNull() &
            col("latitude").between(40.4, 41.0) &
            col("longitude").between(-74.3, -73.6)
        )
    )

    print("Valid NYC coordinates:", geo_df.count())

    # Collect coordinates for geographical analysis.
    # The current dataset contains 100,000 records.
    pdf = geo_df.toPandas()

    if len(pdf) < 8:
        raise ValueError(
            "Not enough valid coordinates for 8 clusters."
        )

    print("Coordinates loaded:", len(pdf))


    # ==========================================
    # 5. K-MEANS CLUSTERING
    # ==========================================

    print("\n3. APPLYING K-MEANS CLUSTERING")

    # Adjust longitude approximately for NYC latitude
    mean_latitude = pdf["latitude"].mean()

    X = pd.DataFrame({
        "latitude": pdf["latitude"],
        "adjusted_longitude": (
            pdf["longitude"] * __import__("math").cos(
                __import__("math").radians(mean_latitude)
            )
        )
    })

    kmeans = KMeans(
        n_clusters=8,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    pdf["cluster_id"] = kmeans.fit_predict(X)

    print("K-Means clustering completed!")

    cluster_summary = (
        pdf.groupby("cluster_id")
        .agg(
            total_complaints=("unique_key", "count"),
            center_latitude=("latitude", "mean"),
            center_longitude=("longitude", "mean")
        )
        .reset_index()
        .sort_values("total_complaints", ascending=False)
    )

    print("\n4. CLUSTER SUMMARY")

    print(cluster_summary.to_string(index=False))

    cluster_summary.to_csv(
        REPORT_DIR / "cluster_summary.csv",
        index=False
    )

    pdf.to_csv(
        REPORT_DIR / "geographical_complaints.csv",
        index=False
    )


    # ==========================================
    # 6. CREATE INTERACTIVE NYC MAP
    # ==========================================

    print("\n5. CREATING INTERACTIVE MAP")

    nyc_map = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=10,
        tiles="CartoDB positron"
    )

    # Heatmap layer: use a sample to keep the map responsive
    heat_sample = pdf.sample(
        n=min(20000, len(pdf)),
        random_state=42
    )

    heat_data = heat_sample[
        ["latitude", "longitude"]
    ].values.tolist()

    heat_layer = folium.FeatureGroup(
        name="Complaint Heatmap"
    )

    HeatMap(
        heat_data,
        radius=12,
        blur=18,
        min_opacity=0.35
    ).add_to(heat_layer)

    heat_layer.add_to(nyc_map)


    # Marker layer: show a smaller sample
    marker_sample = pdf.sample(
        n=min(5000, len(pdf)),
        random_state=42
    )

    marker_data = marker_sample[
        ["latitude", "longitude"]
    ].values.tolist()

    marker_layer = folium.FeatureGroup(
        name="Complaint Locations"
    )

    FastMarkerCluster(
        data=marker_data
    ).add_to(marker_layer)

    marker_layer.add_to(nyc_map)


    # Cluster centers
    center_layer = folium.FeatureGroup(
        name="K-Means Cluster Centers"
    )

    for _, row in cluster_summary.iterrows():

        folium.Marker(
            location=[
                row["center_latitude"],
                row["center_longitude"]
            ],
            popup=(
                f"Cluster: {int(row['cluster_id'])}<br>"
                f"Complaints: {int(row['total_complaints'])}"
            ),
            tooltip=f"Cluster {int(row['cluster_id'])}",
            icon=folium.Icon(
                color="red",
                icon="info-sign"
            )
        ).add_to(center_layer)

    center_layer.add_to(nyc_map)

    folium.LayerControl().add_to(nyc_map)

    map_path = MAP_DIR / "complaint_hotspots.html"

    nyc_map.save(str(map_path))


    # ==========================================
    # 7. FINAL OUTPUT
    # ==========================================

    print("\n" + "=" * 50)
    print("GEOGRAPHICAL ANALYTICS COMPLETED!")
    print("=" * 50)

    print("Total records:", len(pdf))
    print("Number of clusters:", pdf["cluster_id"].nunique())

    print("\nReports saved in:")
    print(REPORT_DIR)

    print("\nInteractive map saved in:")
    print(map_path)

finally:
    spark.stop()
