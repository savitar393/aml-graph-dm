from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def add_basic_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create raw transaction and simple temporal features."""
    out = df.copy()

    if "amount_paid" in out.columns:
        out["amount_paid"] = pd.to_numeric(out["amount_paid"], errors="coerce").fillna(0.0)
    else:
        out["amount_paid"] = 0.0

    if "amount_received" in out.columns:
        out["amount_received"] = pd.to_numeric(out["amount_received"], errors="coerce").fillna(0.0)
    else:
        out["amount_received"] = out["amount_paid"]

    out["amount_delta"] = out["amount_paid"] - out["amount_received"]
    out["log_amount_paid"] = np.log1p(out["amount_paid"].clip(lower=0))
    out["log_amount_received"] = np.log1p(out["amount_received"].clip(lower=0))

    if "from_bank" in out.columns and "to_bank" in out.columns:
        out["is_cross_bank"] = (out["from_bank"].astype(str) != out["to_bank"].astype(str)).astype(int)
    else:
        out["is_cross_bank"] = 0

    if "payment_currency" in out.columns and "receiving_currency" in out.columns:
        out["is_cross_currency"] = (
            out["payment_currency"].astype(str) != out["receiving_currency"].astype(str)
        ).astype(int)
    else:
        out["is_cross_currency"] = 0

    if "timestamp" in out.columns:
        out["hour"] = out["timestamp"].dt.hour.fillna(0).astype(int)
        out["dayofweek"] = out["timestamp"].dt.dayofweek.fillna(0).astype(int)
        out["day"] = out["timestamp"].dt.day.fillna(0).astype(int)
    else:
        out["hour"] = 0
        out["dayofweek"] = 0
        out["day"] = 0

    return out


def make_model_matrix(
    df: pd.DataFrame,
    include_graph_features: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert engineered dataframe to numeric model matrix."""
    if "is_laundering" not in df.columns:
        raise ValueError("Target column 'is_laundering' not found.")

    y = df["is_laundering"].astype(int)

    numeric_features = [
        "amount_paid",
        "amount_received",
        "amount_delta",
        "log_amount_paid",
        "log_amount_received",
        "is_cross_bank",
        "is_cross_currency",
        "hour",
        "dayofweek",
        "day",
    ]

    graph_features = [
        "sender_n_sent",
        "sender_total_sent",
        "sender_unique_receivers",
        "sender_n_received",
        "sender_total_received",
        "sender_unique_senders",
        "receiver_n_sent",
        "receiver_total_sent",
        "receiver_unique_receivers",
        "receiver_n_received",
        "receiver_total_received",
        "receiver_unique_senders",
    ]

    historical_graph_features = [
        "sender_n_sent_prev",
        "sender_total_sent_prev",
        "receiver_n_received_prev",
        "receiver_total_received_prev",
        "pair_n_prev",
        "pair_total_amount_prev",
        "sender_unique_receivers_prev",
        "receiver_unique_senders_prev",
        "fan_out_score",
        "fan_in_score",
        "pair_repeat_score",
        "sender_avg_sent_prev",
        "receiver_avg_received_prev",
        "sender_amount_ratio",
        "receiver_amount_ratio",
        "graph_activity_score",
    ]

    rolling_graph_features = [
        "sender_tx_count_1h_prev",
        "sender_amount_sum_1h_prev",
        "sender_amount_mean_1h_prev",
        "sender_amount_max_1h_prev",
        "receiver_tx_count_1h_prev",
        "receiver_amount_sum_1h_prev",
        "receiver_amount_mean_1h_prev",
        "receiver_amount_max_1h_prev",

        "sender_tx_count_24h_prev",
        "sender_amount_sum_24h_prev",
        "sender_amount_mean_24h_prev",
        "sender_amount_max_24h_prev",
        "receiver_tx_count_24h_prev",
        "receiver_amount_sum_24h_prev",
        "receiver_amount_mean_24h_prev",
        "receiver_amount_max_24h_prev",

        "sender_time_since_last_tx_hours",
        "receiver_time_since_last_tx_hours",

        "rolling_fan_out_score_1h",
        "rolling_fan_in_score_1h",
        "amount_vs_sender_mean_1h",
        "amount_vs_receiver_mean_1h",

        "rolling_fan_out_score_24h",
        "rolling_fan_in_score_24h",
        "amount_vs_sender_mean_24h",
        "amount_vs_receiver_mean_24h",
    ]

    advanced_graph_features = [
        "sender_in_degree_prev",
        "sender_out_degree_prev",
        "receiver_in_degree_prev",
        "receiver_out_degree_prev",

        "sender_in_amount_prev",
        "sender_out_amount_prev",
        "receiver_in_amount_prev",
        "receiver_out_amount_prev",

        "reverse_pair_n_prev",
        "reverse_pair_amount_prev",
        "has_reverse_edge_prev",

        "reciprocal_pair_score",
        "cycle_proxy_score",
        "sender_centrality_balance_prev",
        "receiver_centrality_balance_prev",

        "sender_in_out_degree_ratio_prev",
        "receiver_in_out_degree_ratio_prev",
        "sender_flow_balance_prev",
        "receiver_flow_balance_prev",
    ]

    features: List[str] = [c for c in numeric_features if c in df.columns]
    if include_graph_features:
        features.extend([c for c in graph_features if c in df.columns])
        features.extend([c for c in historical_graph_features if c in df.columns])
        features.extend([c for c in rolling_graph_features if c in df.columns])
        features.extend([c for c in advanced_graph_features if c in df.columns])

    X_num = df[features].fillna(0)

    categorical_candidates = [
        "payment_format",
        "payment_currency",
        "receiving_currency",
    ]
    cat_cols = [c for c in categorical_candidates if c in df.columns]
    if cat_cols:
        X_cat = pd.get_dummies(df[cat_cols].astype(str), dummy_na=True, drop_first=False)
        X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_num.copy()

    return X, y
