from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "reports" / "ml"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

resolution_file = REPORT_DIR / "model_comparison.csv"
delay_file = REPORT_DIR / "delay_model_comparison.csv"

if not resolution_file.exists():
    raise FileNotFoundError(f"Missing report: {resolution_file}")

if not delay_file.exists():
    raise FileNotFoundError(f"Missing report: {delay_file}")

resolution = pd.read_csv(resolution_file)
delay = pd.read_csv(delay_file)

print("\n" + "=" * 65)
print("SMARTCITY311 - CONSOLIDATED MODEL EVALUATION")
print("=" * 65)

print("\n1. RESOLUTION TIME PREDICTION")
print(resolution.to_string(index=False))

required_resolution = {"model", "MAE_hours", "RMSE_hours", "R2"}
if not required_resolution.issubset(resolution.columns):
    raise ValueError("Unexpected resolution report columns.")

best_resolution = resolution.sort_values("MAE_hours").iloc[0]

print("\nBest resolution model:", best_resolution["model"])
print("MAE:", round(best_resolution["MAE_hours"], 3), "hours")
print("RMSE:", round(best_resolution["RMSE_hours"], 3), "hours")
print("R2:", round(best_resolution["R2"], 4))

print("\n2. DELAY CLASSIFICATION")
print(delay.to_string(index=False))

required_delay = {"model", "accuracy", "precision", "recall", "f1_score", "roc_auc"}
if not required_delay.issubset(delay.columns):
    raise ValueError("Unexpected delay report columns.")

best_delay = delay.sort_values("f1_score", ascending=False).iloc[0]

print("\nBest delay classification model:", best_delay["model"])
print("Accuracy:", round(best_delay["accuracy"] * 100, 2), "%")
print("Precision:", round(best_delay["precision"] * 100, 2), "%")
print("Recall:", round(best_delay["recall"] * 100, 2), "%")
print("F1 Score:", round(best_delay["f1_score"] * 100, 2), "%")
print("ROC-AUC:", round(best_delay["roc_auc"], 4))

summary = pd.DataFrame([
    {
        "task": "Resolution Time Prediction",
        "best_model": best_resolution["model"],
        "primary_metric": "MAE_hours",
        "primary_score": best_resolution["MAE_hours"]
    },
    {
        "task": "Delay Classification",
        "best_model": best_delay["model"],
        "primary_metric": "F1 Score",
        "primary_score": best_delay["f1_score"]
    }
])

output_file = REPORT_DIR / "consolidated_model_evaluation.csv"
summary.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("CONSOLIDATED EVALUATION COMPLETED")
print("Saved:", output_file)
print("=" * 65)
