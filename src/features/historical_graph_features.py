from __future__ import annotations

import numpy as np
import pandas as pd


def add_historical_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create leakage-safe graph features.

    Each feature only uses transactions that happened before the current row.
    The dataframe must contain:
    - sender_id
    - receiver_id
    - timestamp
    - amount_paid
    """
    out = df.copy()

    if "timestamp" in out.columns:
        out = out.sort_values("timestamp").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    out["amount_paid"] = pd.to_numeric(out["amount_paid"], errors="coerce").fillna(0.0)

    # Sender historical outgoing behavior
    out["sender_n_sent_prev"] = out.groupby("sender_id").cumcount()

    out["sender_total_sent_prev"] = (
        out.groupby("sender_id")["amount_paid"]
        .cumsum()
        .groupby(out["sender_id"])
        .shift(1)
        .fillna(0.0)
    )

    # Receiver historical incoming behavior
    out["receiver_n_received_prev"] = out.groupby("receiver_id").cumcount()

    out["receiver_total_received_prev"] = (
        out.groupby("receiver_id")["amount_paid"]
        .cumsum()
        .groupby(out["receiver_id"])
        .shift(1)
        .fillna(0.0)
    )

    # Sender historical incoming behavior
    sender_incoming = (
        out.groupby("receiver_id")
        .cumcount()
        .rename("sender_n_received_prev_temp")
    )

    # Receiver historical outgoing behavior
    receiver_outgoing = (
        out.groupby("sender_id")
        .cumcount()
        .rename("receiver_n_sent_prev_temp")
    )

    # Pair historical behavior
    out["pair_n_prev"] = out.groupby(["sender_id", "receiver_id"]).cumcount()

    out["pair_total_amount_prev"] = (
        out.groupby(["sender_id", "receiver_id"])["amount_paid"]
        .cumsum()
        .groupby([out["sender_id"], out["receiver_id"]])
        .shift(1)
        .fillna(0.0)
    )

    # Historical unique receivers for sender
    first_sender_receiver = (
        out.groupby(["sender_id", "receiver_id"])
        .cumcount()
        .eq(0)
        .astype(int)
    )
    out["sender_unique_receivers_prev"] = (
        first_sender_receiver.groupby(out["sender_id"]).cumsum()
        - first_sender_receiver
    )

    # Historical unique senders for receiver
    first_receiver_sender = (
        out.groupby(["receiver_id", "sender_id"])
        .cumcount()
        .eq(0)
        .astype(int)
    )
    out["receiver_unique_senders_prev"] = (
        first_receiver_sender.groupby(out["receiver_id"]).cumsum()
        - first_receiver_sender
    )

    eps = 1e-9

    # AML-inspired graph scores
    out["fan_out_score"] = (
        np.log1p(out["sender_n_sent_prev"])
        * np.log1p(out["sender_unique_receivers_prev"])
    )

    out["fan_in_score"] = (
        np.log1p(out["receiver_n_received_prev"])
        * np.log1p(out["receiver_unique_senders_prev"])
    )

    out["pair_repeat_score"] = np.log1p(out["pair_n_prev"])

    out["sender_avg_sent_prev"] = (
        out["sender_total_sent_prev"]
        / (out["sender_n_sent_prev"] + eps)
    )

    out["receiver_avg_received_prev"] = (
        out["receiver_total_received_prev"]
        / (out["receiver_n_received_prev"] + eps)
    )

    out["sender_amount_ratio"] = (
        out["amount_paid"]
        / (out["sender_avg_sent_prev"] + eps)
    )

    out["receiver_amount_ratio"] = (
        out["amount_paid"]
        / (out["receiver_avg_received_prev"] + eps)
    )

    out["graph_activity_score"] = (
        np.log1p(out["sender_n_sent_prev"])
        + np.log1p(out["receiver_n_received_prev"])
        + np.log1p(out["pair_n_prev"])
    )

    return out