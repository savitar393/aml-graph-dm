from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
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

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


@dataclass
class ModelResult:
    name: str
    model: object
    metrics: dict


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)[:, 1]
    else:
        score = model.decision_function(X_test)

    pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    return {
        "pr_auc": float(average_precision_score(y_test, score)),
        "f1_suspicious": float(f1_score(y_test, pred, zero_division=0)),
        "precision_suspicious": float(precision_score(y_test, pred, zero_division=0)),
        "recall_suspicious": float(recall_score(y_test, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def get_models(scale_pos_weight: float = 1.0) -> Dict[str, object]:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=42,
        )

    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )

    return models


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pos = max(int(y_train.sum()), 1)
    neg = max(len(y_train) - pos, 1)
    scale_pos_weight = neg / pos

    fitted = {}
    rows = []

    for name, model in get_models(scale_pos_weight=scale_pos_weight).items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        fitted[name] = model
        rows.append({"model": name, **metrics})

    metrics_df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    return metrics_df, fitted
