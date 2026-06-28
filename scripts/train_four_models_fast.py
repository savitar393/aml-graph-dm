from __future__ import annotations

from pathlib import Path
import argparse
import joblib

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.data.load_data import chronological_split
from src.features.tabular_features import make_model_matrix


def sample_training_data(X, y, neg_per_pos: int = 100, random_state: int = 42):
    """Keep all positives and sample negatives for faster imbalanced training."""
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_pos = len(pos_idx)
    if n_pos == 0:
        raise ValueError("No positive samples in training data.")

    n_neg = min(len(neg_idx), n_pos * neg_per_pos)
    sampled_neg_idx = neg_idx.to_series().sample(n=n_neg, random_state=random_state).index

    sampled_idx = pos_idx.union(sampled_neg_idx)

    return X.loc[sampled_idx], y.loc[sampled_idx]


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    scores = model.predict_proba(X)[:, 1]
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alert_count": int(pred.sum()),
    }


def get_fast_models(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="saga",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            max_samples=0.8,
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/hi_small_features.parquet")
    parser.add_argument("--neg-per-pos", type=int, default=100)
    parser.add_argument("--include-graph", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading:", data_path)
    df = pd.read_parquet(data_path)

    print("Data shape:", df.shape)
    print("Total labels:")
    print(df["is_laundering"].value_counts())

    train_df, val_df, test_df = chronological_split(df, train_size=0.60, val_size=0.20)

    print("\nSplit label counts:")
    print("Train:")
    print(train_df["is_laundering"].value_counts())
    print("Validation:")
    print(val_df["is_laundering"].value_counts())
    print("Test:")
    print(test_df["is_laundering"].value_counts())

    X_train, y_train = make_model_matrix(train_df, include_graph_features=args.include_graph)
    X_val, y_val = make_model_matrix(val_df, include_graph_features=args.include_graph)
    X_test, y_test = make_model_matrix(test_df, include_graph_features=args.include_graph)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_sample, y_sample = sample_training_data(
        X_train,
        y_train,
        neg_per_pos=args.neg_per_pos,
        random_state=42,
    )

    pos = max(int(y_sample.sum()), 1)
    neg = max(len(y_sample) - pos, 1)
    scale_pos_weight = neg / pos

    print("\nSampled training data:")
    print("Shape:", X_sample.shape)
    print("Positive:", pos)
    print("Negative:", neg)
    print("scale_pos_weight:", scale_pos_weight)

    models = get_fast_models(scale_pos_weight=scale_pos_weight)

    rows = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        model.fit(X_sample, y_sample)

        val_metrics = evaluate_model(model, X_val, y_val, threshold=0.5)
        test_metrics = evaluate_model(model, X_test, y_test, threshold=0.5)

        row = {
            "model": model_name,
            "feature_set": "raw_plus_temporal_graph" if args.include_graph else "raw",
            "neg_per_pos": args.neg_per_pos,
            "feature_count": X_train.shape[1],
            "train_sample_rows": len(X_sample),
            "train_sample_pos": int(y_sample.sum()),
            "train_sample_neg": int(len(y_sample) - y_sample.sum()),
            "val_pr_auc": val_metrics["pr_auc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_tp": test_metrics["tp"],
            "test_fp": test_metrics["fp"],
            "test_tn": test_metrics["tn"],
            "test_fn": test_metrics["fn"],
            "test_alert_count": test_metrics["alert_count"],
        }

        rows.append(row)
        fitted_models[model_name] = model

        print(pd.Series(row))

    result = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    result.to_csv(out_dir / "four_model_comparison_temporal_graph.csv", index=False)

    print("\nFour-model comparison:")
    print(result)

    best_row = result.iloc[0]
    best_model_name = best_row["model"]
    best_model = fitted_models[best_model_name]

    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_set": "raw_plus_graph" if args.include_graph else "raw",
        "feature_columns": list(X_train.columns),
        "threshold": 0.5,
        "selection_metric": "val_pr_auc",
        "val_pr_auc": float(best_row["val_pr_auc"]),
        "test_pr_auc": float(best_row["test_pr_auc"]),
        "training_note": (
            "Best model selected from Logistic Regression, Random Forest, "
            "XGBoost, and LightGBM using validation PR-AUC. "
            "Training used chronological split and negative undersampling."
        ),
    }

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, "models/best_model_bundle.joblib")

    print("\nSaved best model bundle:")
    print("Best model:", best_model_name)
    print("Validation PR-AUC:", best_row["val_pr_auc"])
    print("Test PR-AUC:", best_row["test_pr_auc"])
    print("Saved table:", out_dir / "four_model_comparison_temporal_graph.csv")


if __name__ == "__main__":
    main()