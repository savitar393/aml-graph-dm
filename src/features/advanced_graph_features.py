from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd


def add_temporal_centrality_and_motif_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add leakage-safe temporal centrality and motif/cycle-proxy features.

    Each row uses only transactions that occurred before the current transaction.

    This is not full cycle detection or full community detection.
    It creates interpretable proxy features:
    - previous in/out degree for sender and receiver
    - previous reverse-edge existence
    - previous reciprocal transaction count
    - repeated pair behavior
    - simple cycle-like proxy score
    """
    out = df.copy()

    if "timestamp" not in out.columns:
        raise ValueError("timestamp column is required.")

    required = ["sender_id", "receiver_id", "amount_paid"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = out.sort_values("timestamp").reset_index(drop=True)
    out["amount_paid"] = pd.to_numeric(out["amount_paid"], errors="coerce").fillna(0.0)

    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    in_amount = defaultdict(float)
    out_amount = defaultdict(float)
    pair_count = defaultdict(int)
    pair_amount = defaultdict(float)

    sender_in_degree_prev = []
    sender_out_degree_prev = []
    receiver_in_degree_prev = []
    receiver_out_degree_prev = []

    sender_in_amount_prev = []
    sender_out_amount_prev = []
    receiver_in_amount_prev = []
    receiver_out_amount_prev = []

    reverse_pair_n_prev = []
    reverse_pair_amount_prev = []
    has_reverse_edge_prev = []

    reciprocal_pair_score = []
    cycle_proxy_score = []
    sender_centrality_balance_prev = []
    receiver_centrality_balance_prev = []

    for row in out.itertuples(index=False):
        sender = getattr(row, "sender_id")
        receiver = getattr(row, "receiver_id")
        amount = float(getattr(row, "amount_paid", 0.0))

        s_in = in_degree[sender]
        s_out = out_degree[sender]
        r_in = in_degree[receiver]
        r_out = out_degree[receiver]

        s_in_amt = in_amount[sender]
        s_out_amt = out_amount[sender]
        r_in_amt = in_amount[receiver]
        r_out_amt = out_amount[receiver]

        reverse_n = pair_count[(receiver, sender)]
        reverse_amt = pair_amount[(receiver, sender)]
        pair_n = pair_count[(sender, receiver)]

        sender_in_degree_prev.append(s_in)
        sender_out_degree_prev.append(s_out)
        receiver_in_degree_prev.append(r_in)
        receiver_out_degree_prev.append(r_out)

        sender_in_amount_prev.append(s_in_amt)
        sender_out_amount_prev.append(s_out_amt)
        receiver_in_amount_prev.append(r_in_amt)
        receiver_out_amount_prev.append(r_out_amt)

        reverse_pair_n_prev.append(reverse_n)
        reverse_pair_amount_prev.append(reverse_amt)
        has_reverse_edge_prev.append(int(reverse_n > 0))

        reciprocal_pair_score.append(
            math.log1p(pair_n) * math.log1p(reverse_n)
        )

        # Cycle-like proxy:
        # If receiver has previously sent to sender, and sender has prior incoming activity,
        # this suggests possible circular or reciprocal fund movement.
        cycle_proxy_score.append(
            int(reverse_n > 0)
            * (math.log1p(s_in) + math.log1p(r_out) + math.log1p(reverse_n))
        )

        sender_centrality_balance_prev.append(s_in - s_out)
        receiver_centrality_balance_prev.append(r_in - r_out)

        # Update graph state after extracting features.
        out_degree[sender] += 1
        in_degree[receiver] += 1
        out_amount[sender] += amount
        in_amount[receiver] += amount
        pair_count[(sender, receiver)] += 1
        pair_amount[(sender, receiver)] += amount

    out["sender_in_degree_prev"] = sender_in_degree_prev
    out["sender_out_degree_prev"] = sender_out_degree_prev
    out["receiver_in_degree_prev"] = receiver_in_degree_prev
    out["receiver_out_degree_prev"] = receiver_out_degree_prev

    out["sender_in_amount_prev"] = sender_in_amount_prev
    out["sender_out_amount_prev"] = sender_out_amount_prev
    out["receiver_in_amount_prev"] = receiver_in_amount_prev
    out["receiver_out_amount_prev"] = receiver_out_amount_prev

    out["reverse_pair_n_prev"] = reverse_pair_n_prev
    out["reverse_pair_amount_prev"] = reverse_pair_amount_prev
    out["has_reverse_edge_prev"] = has_reverse_edge_prev

    out["reciprocal_pair_score"] = reciprocal_pair_score
    out["cycle_proxy_score"] = cycle_proxy_score
    out["sender_centrality_balance_prev"] = sender_centrality_balance_prev
    out["receiver_centrality_balance_prev"] = receiver_centrality_balance_prev

    eps = 1e-9

    out["sender_in_out_degree_ratio_prev"] = (
        out["sender_in_degree_prev"] / (out["sender_out_degree_prev"] + eps)
    )

    out["receiver_in_out_degree_ratio_prev"] = (
        out["receiver_in_degree_prev"] / (out["receiver_out_degree_prev"] + eps)
    )

    out["sender_flow_balance_prev"] = (
        out["sender_in_amount_prev"] - out["sender_out_amount_prev"]
    )

    out["receiver_flow_balance_prev"] = (
        out["receiver_in_amount_prev"] - out["receiver_out_amount_prev"]
    )

    return out