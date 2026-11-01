import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

BASE = os.getcwd()
DATA_PATH = os.path.join(BASE, "data", "processed", "nyc_311_cleaned")
MODEL_DIR = os.path.join(BASE, "models")
REPORT_DIR = os.path.join(BASE, "reports", "ml")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("SMARTCITY311 - DELAY CLASSIFICATION")
print("=" * 60)

print("\n[1/6] Loading cleaned dataset...")
df = pd.read_parquet(DATA_PATH)
print("Loaded records:", len(df))

required = [
    "resolution_hours", "agency", "agency_name",
    "complaint_type", "borough", "complaint_hour",
    "complaint_weekday", "complaint_month"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Train only on complaints that have a known resolution duration.
df = df.dropna(subset=["resolution_hours"]).copy()
df = df[df["resolution_hours"] >= 0].copy()

# Project-defined target: more than 24 hours = delayed.
df["is_delayed"] = (df["resolution_hours"] > 24).astype(int)

if df["is_delayed"].nunique() < 2:
    raise ValueError("Dataset must contain both delayed and non-delayed records.")

print("Closed records with valid duration:", len(df))
print("\nTarget distribution:")
print(df["is_delayed"].value_counts().rename(
    index={0: "Not delayed", 1: "Delayed"}
))

print("\n[2/6] Preparing features...")

categorical_features = [
    "agency", "agency_name", "complaint_type", "borough"
]
numeric_features = [
    "complaint_hour", "complaint_weekday", "complaint_month"
]
feature_columns = categorical_features + numeric_features

X = df[feature_columns].copy()
y = df["is_delayed"].copy()

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\n[3/6] Building preprocessing pipeline...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10
                ))
            ]),
            categorical_features
        ),
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        )
    ]
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
}

results = []
trained_models = {}

print("\n[4/6] Training and evaluating models...")

for name, classifier in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    })

    trained_models[name] = pipeline

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")

print("\n[5/6] Saving evaluation report...")

results_df = pd.DataFrame(results).sort_values(
    by="f1_score", ascending=False
)

comparison_path = os.path.join(
    REPORT_DIR, "delay_model_comparison.csv"
)
results_df.to_csv(comparison_path, index=False)

best_name = results_df.iloc[0]["model"]
best_model = trained_models[best_name]

model_path = os.path.join(
    MODEL_DIR, "best_delay_model.joblib"
)
joblib.dump(best_model, model_path, compress=3)

features_path = os.path.join(
    MODEL_DIR, "delay_model_features.json"
)

with open(features_path, "w", encoding="utf-8") as f:
    json.dump({
        "features": feature_columns,
        "target": "is_delayed",
        "target_definition": "resolution_hours > 24",
        "best_model": best_name,
        "threshold_hours": 24,
        "note": "24-hour threshold is project-defined, not an official NYC SLA."
    }, f, indent=4)

print("\n[6/6] Final evaluation...")
best_predictions = best_model.predict(X_test)

print("\nBest model:", best_name)
print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    target_names=["Not delayed", "Delayed"],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, best_predictions))

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nFILES GENERATED:")
print(model_path)
print(comparison_path)
print(features_path)

print("\nDELAY CLASSIFICATION COMPLETED SUCCESSFULLY!")
