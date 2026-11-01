import os
import pandas as pd

print("Loading cleaned NYC 311 data...")
df = pd.read_parquet("data/processed/nyc_311_cleaned")

os.makedirs("reports/department", exist_ok=True)

# 1. Department workload
workload = (
    df.groupby("agency_name", dropna=False)
    .agg(
        total_complaints=("unique_key", "count"),
        closed_complaints=("resolution_status", lambda x: (x == "Closed").sum()),
        open_complaints=("resolution_status", lambda x: (x == "Open").sum())
    )
    .reset_index()
)

workload["closure_percentage"] = (
    workload["closed_complaints"] /
    workload["total_complaints"] * 100
).round(2)

workload = workload.sort_values(
    "total_complaints", ascending=False
)

workload.to_csv(
    "reports/department/department_workload.csv",
    index=False
)

# 2. Department resolution performance
closed = df[
    (df["resolution_status"] == "Closed") &
    (df["resolution_hours"].notna()) &
    (df["resolution_hours"] >= 0)
].copy()

performance = (
    closed.groupby("agency_name")
    .agg(
        closed_sample_count=("resolution_hours", "count"),
        average_resolution_hours=("resolution_hours", "mean"),
        median_resolution_hours=("resolution_hours", "median"),
        p90_resolution_hours=(
            "resolution_hours",
            lambda x: x.quantile(0.90)
        )
    )
    .reset_index()
)

performance = performance[
    performance["closed_sample_count"] >= 30
].copy()

performance = performance.sort_values(
    "average_resolution_hours", ascending=False
)

numeric_cols = [
    "average_resolution_hours",
    "median_resolution_hours",
    "p90_resolution_hours"
]

performance[numeric_cols] = performance[numeric_cols].round(2)

performance.to_csv(
    "reports/department/department_resolution_performance.csv",
    index=False
)

print("\nDEPARTMENT REPORTS REGENERATED!")
print("\nWorkload report columns:")
print(pd.read_csv("reports/department/department_workload.csv").columns.tolist())

print("\nPerformance report columns:")
print(pd.read_csv("reports/department/department_resolution_performance.csv").columns.tolist())

print("\nDepartment workload preview:")
print(workload.head(5).to_string(index=False))

print("\nDepartment performance preview:")
print(performance.head(5).to_string(index=False))

print("\nSUCCESS!")
