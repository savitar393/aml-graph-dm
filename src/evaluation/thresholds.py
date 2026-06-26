from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def find_best_threshold(y_true, scores, metric: str = "f1") -> dict:
    """Find best threshold using validation/test scores.

    For AML, threshold 0.5 is usually not optimal because the positive class is rare.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    for t in thresholds:
        pred = (scores >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "alert_count": int(pred.sum()),
        })

    df = pd.DataFrame(rows)

    if metric == "f1":
        best = df.sort_values("f1", ascending=False).iloc[0]
    elif metric == "recall":
        best = df.sort_values(["recall", "precision"], ascending=False).iloc[0]
    else:
        raise ValueError("metric must be either 'f1' or 'recall'")

    return {
        "best_threshold": float(best["threshold"]),
        "best_f1": float(best["f1"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "alert_count": int(best["alert_count"]),
        "threshold_table": df,
    }


def precision_at_k(y_true, scores, k: int) -> float:
    temp = pd.DataFrame({"y": y_true, "score": scores})
    temp = temp.sort_values("score", ascending=False).head(k)

    if len(temp) == 0:
        return 0.0

    return float(temp["y"].mean())


def recall_at_k(y_true, scores, k: int) -> float:
    temp = pd.DataFrame({"y": y_true, "score": scores})
    total_positive = max(int(temp["y"].sum()), 1)

    top_k = temp.sort_values("score", ascending=False).head(k)
    return float(top_k["y"].sum() / total_positive)