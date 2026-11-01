from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "processed" / "nyc_311_cleaned"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports" / "ml"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("SmartCity311_ResolutionPrediction")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

try:
    df = spark.read.parquet(str(DATA))

    print("\nDATASET SCHEMA")
    df.printSchema()

    pdf = df.toPandas()

    print("\nTotal records:", len(pdf))

    pdf["created_date"] = pd.to_datetime(
        pdf["created_date"], errors="coerce"
    )

    pdf["resolution_hours"] = pd.to_numeric(
        pdf["resolution_hours"], errors="coerce"
    )

    # Only completed complaints with valid non-negative duration
    pdf = pdf[
        (pdf["resolution_status"] == "Closed") &
        pdf["resolution_hours"].notna() &
        (pdf["resolution_hours"] >= 0) &
        pdf["created_date"].notna()
    ].copy()

    print("Valid closed records:", len(pdf))

    if len(pdf) < 100:
        raise ValueError("Not enough valid closed records for model training.")

    # Features available at complaint creation time
    pdf["created_hour"] = pdf["created_date"].dt.hour
    pdf["created_weekday"] = pdf["created_date"].dt.dayofweek
    pdf["created_month"] = pdf["created_date"].dt.month

    categorical = [
        "agency",
        "agency_name",
        "complaint_type",
        "borough"
    ]

    numeric = [
        "created_hour",
        "created_weekday",
        "created_month"
    ]

    # Do not use closed_date, resolution_status,
    # resolution_hours, or unique_key as predictors.
    X = pdf[categorical + numeric].copy()
    y = pdf["resolution_hours"].copy()

    for c in categorical:
        X[c] = X[c].fillna("Unknown").astype(str)

    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric
            )
        ]
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    results = []
    trained = {}

    print("\nTRAINING AND EVALUATION")

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "MAE_hours": mae,
            "RMSE_hours": rmse,
            "R2": r2
        })

        trained[name] = pipeline

        print(
            f"{name}: MAE={mae:.3f}, "
            f"RMSE={rmse:.3f}, R2={r2:.4f}"
        )

    result_df = pd.DataFrame(results).sort_values(
        "MAE_hours", ascending=True
    )

    result_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    best_name = result_df.iloc[0]["model"]
    best_model = trained[best_name]

    joblib.dump(
        best_model,
        MODEL_DIR / "best_resolution_model.joblib"
    )

    with open(MODEL_DIR / "model_features.json", "w") as f:
        json.dump({
            "categorical": categorical,
            "numeric": numeric,
            "target": "resolution_hours",
            "best_model": best_name
        }, f, indent=4)

    print("\nMODEL COMPARISON")
    print(result_df.to_string(index=False))

    print("\nBEST MODEL:", best_name)
    print("Saved:", MODEL_DIR / "best_resolution_model.joblib")
    print("Report:", REPORT_DIR / "model_comparison.csv")
    print("\nRESOLUTION PREDICTION TRAINING COMPLETED!")

finally:
    spark.stop()
