from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.data.load_data import chronological_split
from src.features.tabular_features import make_model_matrix
from src.evaluation.thresholds import find_best_threshold, top_k_table


DATA_PATH = Path("data/processed/hi_small_features.parquet")
MODEL_PATH = Path("models/best_model_bundle.joblib")
OUT_DIR = Path("reports/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processed data...")
df = pd.read_parquet(DATA_PATH)

print("Splitting chronologically...")
train_df, val_df, test_df = chronological_split(df, train_size=0.60, val_size=0.20)

print("Loading model bundle...")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_set = bundle.get("feature_set", "raw_plus_graph")
feature_columns = bundle["feature_columns"]

include_graph = feature_set == "raw_plus_graph"

print("Building validation/test matrices...")
X_val, y_val = make_model_matrix(val_df, include_graph_features=include_graph)
X_test, y_test = make_model_matrix(test_df, include_graph_features=include_graph)

X_val = X_val.reindex(columns=feature_columns, fill_value=0)
X_test = X_test.reindex(columns=feature_columns, fill_value=0)

print("Scoring...")
val_scores = model.predict_proba(X_val)[:, 1]
test_scores = model.predict_proba(X_test)[:, 1]

print("Finding best threshold on validation set...")
threshold_result = find_best_threshold(y_val, val_scores, metric="f1")
best_threshold = threshold_result["best_threshold"]

test_pred_default = (test_scores >= 0.5).astype(int)
test_pred_tuned = (test_scores >= best_threshold).astype(int)

final_eval = pd.DataFrame([
    {
        "setting": "default_threshold_0.5",
        "threshold": 0.5,
        "pr_auc": average_precision_score(y_test, test_scores),
        "f1": f1_score(y_test, test_pred_default, zero_division=0),
        "precision": precision_score(y_test, test_pred_default, zero_division=0),
        "recall": recall_score(y_test, test_pred_default, zero_division=0),
        "alert_count": int(test_pred_default.sum()),
    },
    {
        "setting": "tuned_threshold_validation",
        "threshold": best_threshold,
        "pr_auc": average_precision_score(y_test, test_scores),
        "f1": f1_score(y_test, test_pred_tuned, zero_division=0),
        "precision": precision_score(y_test, test_pred_tuned, zero_division=0),
        "recall": recall_score(y_test, test_pred_tuned, zero_division=0),
        "alert_count": int(test_pred_tuned.sum()),
    },
])

topk = top_k_table(y_test, test_scores, k_values=(50, 100, 200, 500, 1000))

cm = confusion_matrix(y_test, test_pred_tuned, labels=[0, 1])
confusion_df = pd.DataFrame(
    cm,
    index=["actual_normal", "actual_laundering"],
    columns=["predicted_normal", "predicted_laundering"],
)

threshold_table = threshold_result["threshold_table"]

final_eval.to_csv(OUT_DIR / "final_threshold_evaluation.csv", index=False)
topk.to_csv(OUT_DIR / "top_k_evaluation.csv", index=False)
confusion_df.to_csv(OUT_DIR / "confusion_matrix_tuned.csv")
threshold_table.to_csv(OUT_DIR / "threshold_search_table.csv", index=False)

bundle["threshold"] = float(best_threshold)
bundle["validation_threshold_result"] = {
    "best_threshold": float(threshold_result["best_threshold"]),
    "best_f1": float(threshold_result["best_f1"]),
    "best_precision": float(threshold_result["best_precision"]),
    "best_recall": float(threshold_result["best_recall"]),
    "best_alert_count": int(
        threshold_result.get("best_alert_count", threshold_result.get("alert_count", 0))
    ),  
}

joblib.dump(bundle, MODEL_PATH)

print("\nFinal threshold evaluation:")
print(final_eval)

print("\nTop-k evaluation:")
print(topk)

print("\nTuned confusion matrix:")
print(confusion_df)

print("\nUpdated model bundle threshold:", best_threshold)
print("Saved:")
print(OUT_DIR / "final_threshold_evaluation.csv")
print(OUT_DIR / "top_k_evaluation.csv")
print(OUT_DIR / "confusion_matrix_tuned.csv")
print(OUT_DIR / "threshold_search_table.csv")