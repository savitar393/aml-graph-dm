from __future__ import annotations

from pathlib import Path
import argparse
import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from lightgbm import LGBMClassifier

from src.data.load_data import chronological_split
from src.features.tabular_features import make_model_matrix


def sample_training_data(X, y, neg_per_pos: int = 50, random_state: int = 42):
    """Keep all positives and sample negatives for faster imbalanced training."""
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_pos = len(pos_idx)
    n_neg = min(len(neg_idx), n_pos * neg_per_pos)

    if n_pos == 0:
        raise ValueError("No positive laundering samples in training split.")

    sampled_neg_idx = neg_idx.to_series().sample(n=n_neg, random_state=random_state).index
    sampled_idx = pos_idx.union(sampled_neg_idx)

    return X.loc[sampled_idx], y.loc[sampled_idx]


def evaluate_scores(y_true, scores, threshold: float = 0.5):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "pr_auc": average_precision_score(y_true, scores),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alert_count": int(pred.sum()),
    }


def train_lgbm(X_train, y_train, neg_per_pos: int):
    X_sample, y_sample = sample_training_data(
        X_train,
        y_train,
        neg_per_pos=neg_per_pos,
        random_state=42,
    )

    pos = max(int(y_sample.sum()), 1)
    neg = max(len(y_sample) - pos, 1)
    scale_pos_weight = neg / pos

    print(f"Sampled train shape: {X_sample.shape}")
    print(f"Sampled positives: {pos}")
    print(f"Sampled negatives: {neg}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_sample, y_sample)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/hi_small_features.parquet")
    parser.add_argument("--neg-per-pos", type=int, default=50)
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

    experiments = [
        ("raw", False),
        ("raw_plus_graph", True),
    ]

    rows = []
    fitted = {}

    for feature_set, include_graph in experiments:
        print(f"\n=== Training LightGBM: {feature_set} ===")

        X_train, y_train = make_model_matrix(train_df, include_graph_features=include_graph)
        X_val, y_val = make_model_matrix(val_df, include_graph_features=include_graph)
        X_test, y_test = make_model_matrix(test_df, include_graph_features=include_graph)

        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        model = train_lgbm(X_train, y_train, neg_per_pos=args.neg_per_pos)

        val_scores = model.predict_proba(X_val)[:, 1]
        test_scores = model.predict_proba(X_test)[:, 1]

        val_metrics = evaluate_scores(y_val, val_scores, threshold=0.5)
        test_metrics = evaluate_scores(y_test, test_scores, threshold=0.5)

        row = {
            "model": "lightgbm",
            "feature_set": feature_set,
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
            "feature_count": X_train.shape[1],
        }

        rows.append(row)
        fitted[feature_set] = {
            "model": model,
            "feature_columns": list(X_train.columns),
        }

        print(pd.Series(row))

    result = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    result.to_csv(out_dir / "lgbm_fast_metrics.csv", index=False)

    print("\nFast LightGBM metrics:")
    print(result)

    best = result.iloc[0]
    best_feature_set = best["feature_set"]

    bundle = {
        "model": fitted[best_feature_set]["model"],
        "model_name": "lightgbm",
        "feature_set": best_feature_set,
        "feature_columns": fitted[best_feature_set]["feature_columns"],
        "threshold": 0.5,
        "selection_metric": "val_pr_auc",
        "val_pr_auc": float(best["val_pr_auc"]),
        "test_pr_auc": float(best["test_pr_auc"]),
        "training_note": (
            "LightGBM trained with chronological split and negative undersampling "
            "on training data only."
        ),
    }

    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, "models/best_model_bundle.joblib")

    print("\nSaved best model bundle:")
    print("Feature set:", best_feature_set)
    print("Validation PR-AUC:", best["val_pr_auc"])
    print("Test PR-AUC:", best["test_pr_auc"])


if __name__ == "__main__":
    main()
