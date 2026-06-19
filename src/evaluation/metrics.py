from __future__ import annotations

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def precision_at_k(y_true, scores, k: int) -> float:
    df = pd.DataFrame({"y": y_true, "score": scores}).sort_values("score", ascending=False)
    top = df.head(k)
    if len(top) == 0:
        return 0.0
    return float(top["y"].mean())


def recall_at_k(y_true, scores, k: int) -> float:
    df = pd.DataFrame({"y": y_true, "score": scores}).sort_values("score", ascending=False)
    total_pos = max(int(df["y"].sum()), 1)
    return float(df.head(k)["y"].sum() / total_pos)
