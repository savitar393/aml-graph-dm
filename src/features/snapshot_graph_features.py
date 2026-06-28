from __future__ import annotations

from pathlib import Path
import warnings

import networkx as nx
import numpy as np
import pandas as pd


def _safe_window_name(window: str) -> str:
    return window.lower().replace(" ", "").replace("hours", "h").replace("hour", "h")


def _build_weighted_digraph(
    hist_df: pd.DataFrame,
    max_edges: int = 300_000,
) -> nx.DiGraph:
    """Build weighted directed graph from historical transactions only."""
    if hist_df.empty:
        return nx.DiGraph()

    edges = (
        hist_df.groupby(["sender_id", "receiver_id"])
        .agg(
            weight=("amount_paid", "sum"),
            tx_count=("amount_paid", "size"),
        )
        .reset_index()
    )

    # Keep graph computation bounded. Prefer edges with highest total flow.
    if len(edges) > max_edges:
        edges = edges.sort_values("weight", ascending=False).head(max_edges)

    G = nx.DiGraph()

    for row in edges.itertuples(index=False):
        G.add_edge(
            row.sender_id,
            row.receiver_id,
            weight=float(row.weight),
            tx_count=int(row.tx_count),
        )

    return G


def _safe_pagerank(G: nx.DiGraph) -> dict:
    """Compute PageRank with a fallback if convergence fails."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {}

    try:
        return nx.pagerank(
            G,
            alpha=0.85,
            max_iter=50,
            tol=1e-4,
            weight="weight",
        )
    except Exception as exc:
        warnings.warn(f"PageRank failed, falling back to weighted degree: {exc}")

        degree = dict(G.degree(weight="weight"))
        total = sum(degree.values()) or 1.0
        return {node: value / total for node, value in degree.items()}


def _component_features(G: nx.DiGraph) -> tuple[dict, dict]:
    """
    Compute weak-component/community proxy features.

    This is intentionally not full Louvain community detection.
    It gives leakage-safe community-like features:
    - component id
    - component size
    """
    if G.number_of_nodes() == 0:
        return {}, {}

    UG = G.to_undirected()
    component_id = {}
    component_size = {}

    for cid, comp in enumerate(nx.connected_components(UG)):
        size = len(comp)
        for node in comp:
            component_id[node] = cid
            component_size[node] = size

    return component_id, component_size


def add_snapshot_pagerank_community_features(
    df: pd.DataFrame,
    block_freq: str = "1D",
    history_window: str = "24h",
    max_edges_per_snapshot: int = 300_000,
) -> pd.DataFrame:
    """
    Add leakage-safe PageRank and community/component snapshot features.

    For each time block, features are computed only from transactions before
    the current block and inside the selected historical window.

    Example:
    - block_freq="1D"
    - history_window="24h"

    means each day's transactions receive PageRank/community features computed
    from the previous 24 hours of transactions only.
    """
    out = df.copy()

    required = ["timestamp", "sender_id", "receiver_id", "amount_paid"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = out.sort_values("timestamp").reset_index(drop=True)
    out["amount_paid"] = pd.to_numeric(out["amount_paid"], errors="coerce").fillna(0.0)

    wname = _safe_window_name(history_window)

    sender_pr_col = f"sender_pagerank_{wname}_prev"
    receiver_pr_col = f"receiver_pagerank_{wname}_prev"

    sender_comp_size_col = f"sender_community_size_{wname}_prev"
    receiver_comp_size_col = f"receiver_community_size_{wname}_prev"
    same_comp_col = f"same_community_{wname}_prev"

    sender_comp_seen_col = f"sender_seen_in_snapshot_{wname}"
    receiver_comp_seen_col = f"receiver_seen_in_snapshot_{wname}"

    for col in [
        sender_pr_col,
        receiver_pr_col,
        sender_comp_size_col,
        receiver_comp_size_col,
        same_comp_col,
        sender_comp_seen_col,
        receiver_comp_seen_col,
    ]:
        out[col] = 0.0

    out["_snapshot_block"] = out["timestamp"].dt.floor(block_freq)
    blocks = sorted(out["_snapshot_block"].dropna().unique())

    window_delta = pd.Timedelta(history_window)

    for i, block_start in enumerate(blocks, start=1):
        block_start = pd.Timestamp(block_start)
        hist_start = block_start - window_delta

        current_mask = out["_snapshot_block"] == block_start
        hist_mask = (out["timestamp"] < block_start) & (out["timestamp"] >= hist_start)

        hist_df = out.loc[hist_mask, ["sender_id", "receiver_id", "amount_paid"]]
        current_idx = out.index[current_mask]

        print(
            f"[{i}/{len(blocks)}] block={block_start} "
            f"current_rows={len(current_idx):,} hist_rows={len(hist_df):,}"
        )

        if len(current_idx) == 0:
            continue

        G = _build_weighted_digraph(hist_df, max_edges=max_edges_per_snapshot)
        pr = _safe_pagerank(G)
        component_id, component_size = _component_features(G)

        current_sender = out.loc[current_idx, "sender_id"]
        current_receiver = out.loc[current_idx, "receiver_id"]

        sender_pr = current_sender.map(pr).fillna(0.0)
        receiver_pr = current_receiver.map(pr).fillna(0.0)

        sender_comp_id = current_sender.map(component_id)
        receiver_comp_id = current_receiver.map(component_id)

        sender_comp_size = current_sender.map(component_size).fillna(0.0)
        receiver_comp_size = current_receiver.map(component_size).fillna(0.0)

        same_component = (
            sender_comp_id.notna()
            & receiver_comp_id.notna()
            & (sender_comp_id == receiver_comp_id)
        ).astype(int)

        sender_seen = sender_comp_id.notna().astype(int)
        receiver_seen = receiver_comp_id.notna().astype(int)

        out.loc[current_idx, sender_pr_col] = sender_pr.to_numpy()
        out.loc[current_idx, receiver_pr_col] = receiver_pr.to_numpy()

        out.loc[current_idx, sender_comp_size_col] = sender_comp_size.to_numpy()
        out.loc[current_idx, receiver_comp_size_col] = receiver_comp_size.to_numpy()
        out.loc[current_idx, same_comp_col] = same_component.to_numpy()

        out.loc[current_idx, sender_comp_seen_col] = sender_seen.to_numpy()
        out.loc[current_idx, receiver_comp_seen_col] = receiver_seen.to_numpy()

    out = out.drop(columns=["_snapshot_block"])

    return out