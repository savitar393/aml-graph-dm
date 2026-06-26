from __future__ import annotations

from pathlib import Path
import pandas as pd
from pyvis.network import Network


def get_local_transactions(
    df: pd.DataFrame,
    selected_idx: int,
    time_window_hours: int = 24,
    max_edges: int = 150,
) -> tuple[pd.DataFrame, dict]:
    """Return transactions around the selected transaction's sender and receiver."""
    selected = df.loc[selected_idx]

    sender = selected["sender_id"]
    receiver = selected["receiver_id"]

    local_df = df.copy()

    if "timestamp" in local_df.columns and pd.notna(selected.get("timestamp")):
        ts = selected["timestamp"]
        start = ts - pd.Timedelta(hours=time_window_hours)
        end = ts + pd.Timedelta(hours=time_window_hours)
        local_df = local_df[local_df["timestamp"].between(start, end)]

    one_hop_accounts = {sender, receiver}

    local_df = local_df[
        local_df["sender_id"].isin(one_hop_accounts)
        | local_df["receiver_id"].isin(one_hop_accounts)
    ].copy()

    # Prefer high-score or high-amount edges when limiting graph size.
    if "suspicious_score" in local_df.columns:
        local_df = local_df.sort_values("suspicious_score", ascending=False)
    elif "amount_paid" in local_df.columns:
        local_df = local_df.sort_values("amount_paid", ascending=False)

    local_df = local_df.head(max_edges)

    selected_info = {
        "sender": sender,
        "receiver": receiver,
        "amount": selected.get("amount_paid", None),
        "timestamp": selected.get("timestamp", None),
        "score": selected.get("suspicious_score", None),
    }

    return local_df, selected_info


def aggregate_edges_for_visualization(local_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated sender→receiver edges for readability."""
    amount_col = "amount_paid" if "amount_paid" in local_df.columns else None

    agg_dict = {
        "edge_count": ("sender_id", "size"),
    }

    if amount_col:
        agg_dict["total_amount"] = (amount_col, "sum")

    if "suspicious_score" in local_df.columns:
        agg_dict["max_score"] = ("suspicious_score", "max")

    if "is_laundering" in local_df.columns:
        agg_dict["laundering_count"] = ("is_laundering", "sum")

    edge_df = (
        local_df
        .groupby(["sender_id", "receiver_id"])
        .agg(**agg_dict)
        .reset_index()
    )

    if "total_amount" not in edge_df.columns:
        edge_df["total_amount"] = 0.0

    if "max_score" not in edge_df.columns:
        edge_df["max_score"] = 0.0

    if "laundering_count" not in edge_df.columns:
        edge_df["laundering_count"] = 0

    return edge_df


def build_pyvis_graph(
    local_df: pd.DataFrame,
    selected_sender: str,
    selected_receiver: str,
    output_path: str | Path,
) -> Path:
    """Build and save an interactive PyVis graph."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    edge_df = aggregate_edges_for_visualization(local_df)

    net = Network(
        height="650px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
    )

    accounts = set(edge_df["sender_id"]).union(set(edge_df["receiver_id"]))

    # Degree-like node size from local graph participation
    local_degree = {}
    for _, row in edge_df.iterrows():
        local_degree[row["sender_id"]] = local_degree.get(row["sender_id"], 0) + row["edge_count"]
        local_degree[row["receiver_id"]] = local_degree.get(row["receiver_id"], 0) + row["edge_count"]

    for account in accounts:
        is_selected_node = account in {selected_sender, selected_receiver}

        color = "#f59e0b" if is_selected_node else "#9ca3af"
        size = 25 if is_selected_node else min(10 + local_degree.get(account, 1), 35)

        net.add_node(
            account,
            label=str(account),
            title=f"Account: {account}<br>Local degree: {local_degree.get(account, 0)}",
            color=color,
            size=size,
        )

    for _, row in edge_df.iterrows():
        is_selected_edge = (
            row["sender_id"] == selected_sender
            and row["receiver_id"] == selected_receiver
        )

        color = "#dc2626" if is_selected_edge else "#6b7280"
        width = 5 if is_selected_edge else min(1 + row["edge_count"], 6)

        title = (
            f"From: {row['sender_id']}<br>"
            f"To: {row['receiver_id']}<br>"
            f"Transactions: {int(row['edge_count'])}<br>"
            f"Total amount: {row['total_amount']:.2f}<br>"
            f"Max score: {row['max_score']:.4f}<br>"
            f"Laundering labels in local data: {int(row['laundering_count'])}"
        )

        net.add_edge(
            row["sender_id"],
            row["receiver_id"],
            title=title,
            color=color,
            width=width,
            arrows="to",
        )

    net.set_options("""
    var options = {
      "nodes": {
        "borderWidth": 1,
        "font": {
          "size": 12
        }
      },
      "edges": {
        "smooth": {
          "type": "dynamic"
        },
        "font": {
          "size": 10
        }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 120,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      }
    }
    """)

    net.save_graph(str(output_path))
    return output_path