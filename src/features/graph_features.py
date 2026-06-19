from __future__ import annotations

import pandas as pd
import networkx as nx


def add_account_graph_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add scalable graph-style aggregate features using pandas groupby.

    These features approximate directed graph behavior without requiring
    NetworkX over the full million-row graph.
    """
    out = df.copy()

    amount_col = "amount_paid" if "amount_paid" in out.columns else None
    if amount_col:
        out[amount_col] = pd.to_numeric(out[amount_col], errors="coerce").fillna(0.0)
    else:
        out["amount_paid"] = 0.0
        amount_col = "amount_paid"

    sent = (
        out.groupby("sender_id")
        .agg(
            n_sent=("receiver_id", "size"),
            total_sent=(amount_col, "sum"),
            unique_receivers=("receiver_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"sender_id": "account_id"})
    )

    received = (
        out.groupby("receiver_id")
        .agg(
            n_received=("sender_id", "size"),
            total_received=(amount_col, "sum"),
            unique_senders=("sender_id", "nunique"),
        )
        .reset_index()
        .rename(columns={"receiver_id": "account_id"})
    )

    account_features = sent.merge(received, on="account_id", how="outer").fillna(0)

    sender_features = account_features.add_prefix("sender_").rename(columns={"sender_account_id": "sender_id"})
    receiver_features = account_features.add_prefix("receiver_").rename(columns={"receiver_account_id": "receiver_id"})

    out = out.merge(sender_features, on="sender_id", how="left")
    out = out.merge(receiver_features, on="receiver_id", how="left")

    feature_cols = [c for c in out.columns if c.startswith("sender_") or c.startswith("receiver_")]
    out[feature_cols] = out[feature_cols].fillna(0)
    return out


def build_sample_graph(df: pd.DataFrame, max_edges: int = 50_000) -> nx.DiGraph:
    """Build a directed NetworkX graph from a sample for visualization/statistics."""
    sample = df.head(max_edges).copy()

    G = nx.DiGraph()
    for row in sample.itertuples(index=False):
        sender = getattr(row, "sender_id")
        receiver = getattr(row, "receiver_id")
        amount = getattr(row, "amount_paid", 0.0)
        label = getattr(row, "is_laundering", 0)
        G.add_edge(sender, receiver, amount=amount, is_laundering=int(label))
    return G


def graph_summary(G: nx.DiGraph) -> dict:
    """Return basic graph statistics."""
    if G.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0,
            "weak_components": 0,
            "largest_weak_component": 0,
        }

    weak_components = list(nx.weakly_connected_components(G))
    largest = max(len(c) for c in weak_components) if weak_components else 0

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "weak_components": len(weak_components),
        "largest_weak_component": largest,
        "avg_in_degree": sum(dict(G.in_degree()).values()) / G.number_of_nodes(),
        "avg_out_degree": sum(dict(G.out_degree()).values()) / G.number_of_nodes(),
    }
