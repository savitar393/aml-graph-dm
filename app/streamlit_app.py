from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

temp_path = PROJECT_ROOT / "data/sample/uploaded_transactions.csv"

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data.load_data import load_transactions
from src.features.tabular_features import add_basic_transaction_features, make_model_matrix
from src.features.graph_features import add_account_graph_aggregate_features
from src.features.historical_graph_features import add_historical_graph_features
from src.features.rolling_graph_features import add_rolling_graph_features
from src.features.advanced_graph_features import add_temporal_centrality_and_motif_features
from src.visualization.graph_viz import get_local_transactions, build_pyvis_graph


def make_node2vec_embedding_features(df, embeddings, dimensions):
    import numpy as np
    import pandas as pd

    sender_arr = np.zeros((len(df), dimensions), dtype=np.float32)
    receiver_arr = np.zeros((len(df), dimensions), dtype=np.float32)

    sender_known = np.zeros(len(df), dtype=np.int8)
    receiver_known = np.zeros(len(df), dtype=np.int8)

    senders = df["sender_id"].astype(str).to_numpy()
    receivers = df["receiver_id"].astype(str).to_numpy()

    for i, account in enumerate(senders):
        emb = embeddings.get(account)
        if emb is not None:
            sender_arr[i] = emb
            sender_known[i] = 1

    for i, account in enumerate(receivers):
        emb = embeddings.get(account)
        if emb is not None:
            receiver_arr[i] = emb
            receiver_known[i] = 1

    data = {}

    for j in range(dimensions):
        data[f"sender_n2v_{j}"] = sender_arr[:, j]
        data[f"receiver_n2v_{j}"] = receiver_arr[:, j]

    data["sender_n2v_known"] = sender_known
    data["receiver_n2v_known"] = receiver_known

    return pd.DataFrame(data)


def build_explanation_table(row: pd.Series) -> pd.DataFrame:
    """Create a small explanation table for the selected transaction."""
    feature_labels = {
        "amount_paid": "Transaction amount paid",
        "amount_received": "Transaction amount received",
        "amount_delta": "Paid - received amount difference",
        "log_amount_paid": "Log amount paid",
        "is_cross_bank": "Cross-bank transaction",
        "is_cross_currency": "Cross-currency transaction",

        "sender_n_sent": "Sender total outgoing transactions",
        "sender_total_sent": "Sender total outgoing amount",
        "sender_unique_receivers": "Sender unique receivers",
        "sender_n_received": "Sender total incoming transactions",
        "sender_total_received": "Sender total incoming amount",
        "sender_unique_senders": "Sender unique senders",

        "receiver_n_sent": "Receiver total outgoing transactions",
        "receiver_total_sent": "Receiver total outgoing amount",
        "receiver_unique_receivers": "Receiver unique receivers",
        "receiver_n_received": "Receiver total incoming transactions",
        "receiver_total_received": "Receiver total incoming amount",
        "receiver_unique_senders": "Receiver unique senders",
        
        "sender_n_sent_prev": "Previous outgoing transactions by sender",
        "sender_total_sent_prev": "Previous total amount sent by sender",
        "receiver_n_received_prev": "Previous incoming transactions to receiver",
        "receiver_total_received_prev": "Previous total amount received by receiver",
        "pair_n_prev": "Previous repeated transactions from sender to receiver",
        "pair_total_amount_prev": "Previous total amount from sender to receiver",
        "sender_unique_receivers_prev": "Previous unique receivers of sender",
        "receiver_unique_senders_prev": "Previous unique senders to receiver",
        "fan_out_score": "Fan-out behavior score",
        "fan_in_score": "Fan-in behavior score",
        "pair_repeat_score": "Repeated-pair behavior score",
        "sender_amount_ratio": "Current amount compared to sender historical average",
        "receiver_amount_ratio": "Current amount compared to receiver historical average",
        "graph_activity_score": "Combined historical graph activity score",

        "sender_tx_count_1h_prev": "Sender transactions in previous 1 hour",
        "sender_amount_sum_1h_prev": "Sender total amount in previous 1 hour",
        "receiver_tx_count_1h_prev": "Receiver transactions in previous 1 hour",
        "receiver_amount_sum_1h_prev": "Receiver total received in previous 1 hour",

        "sender_tx_count_24h_prev": "Sender transactions in previous 24 hours",
        "sender_amount_sum_24h_prev": "Sender total amount in previous 24 hours",
        "receiver_tx_count_24h_prev": "Receiver transactions in previous 24 hours",
        "receiver_amount_sum_24h_prev": "Receiver total received in previous 24 hours",

        "sender_time_since_last_tx_hours": "Hours since sender's previous transaction",
        "receiver_time_since_last_tx_hours": "Hours since receiver's previous transaction",

        "rolling_fan_out_score_1h": "Short-term fan-out score, 1 hour",
        "rolling_fan_in_score_1h": "Short-term fan-in score, 1 hour",
        "rolling_fan_out_score_24h": "Short-term fan-out score, 24 hours",
        "rolling_fan_in_score_24h": "Short-term fan-in score, 24 hours",

        "sender_in_degree_prev": "Previous incoming degree of sender",
        "sender_out_degree_prev": "Previous outgoing degree of sender",
        "receiver_in_degree_prev": "Previous incoming degree of receiver",
        "receiver_out_degree_prev": "Previous outgoing degree of receiver",
        "reverse_pair_n_prev": "Previous reverse-direction transactions",
        "reverse_pair_amount_prev": "Previous reverse-direction total amount",
        "has_reverse_edge_prev": "Whether reverse transaction existed before",
        "reciprocal_pair_score": "Reciprocal transaction behavior score",
        "cycle_proxy_score": "Cycle-like transaction proxy score",
        "sender_centrality_balance_prev": "Sender previous in-degree minus out-degree",
        "receiver_centrality_balance_prev": "Receiver previous in-degree minus out-degree",
        "sender_flow_balance_prev": "Sender previous inflow minus outflow",
        "receiver_flow_balance_prev": "Receiver previous inflow minus outflow",
        "sender_n2v_known": "Sender account has node2vec embedding",
        "receiver_n2v_known": "Receiver account has node2vec embedding",
    }

    rows = []

    for feature, label in feature_labels.items():
        if feature in row.index:
            value = row[feature]

            rows.append({
                "Feature": feature,
                "Meaning": label,
                "Value": value,
            })

    return pd.DataFrame(rows)


st.set_page_config(page_title="AML Graph Scoring Demo", layout="wide")

st.title("AML Graph-Enhanced Scoring Dashboard")
st.write(
    "This dashboard scores transactions using the trained AML model and shows "
    "a local transaction graph around selected suspicious alerts."
)

node2vec_path = PROJECT_ROOT / "models/node2vec_model_bundle.joblib"
default_path = PROJECT_ROOT / "models/best_model_bundle.joblib"

if node2vec_path.exists():
    model_path = node2vec_path
else:
    model_path = default_path

bundle = joblib.load(model_path)
model = bundle["model"]
feature_columns = bundle["feature_columns"]
feature_set = bundle.get("feature_set", "raw_plus_graph")
threshold = bundle.get("threshold", 0.5)

st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload transaction CSV", type=["csv"])
top_n = st.sidebar.slider("Top suspicious transactions", min_value=10, max_value=200, value=50)
score_threshold = st.sidebar.slider("Score threshold", 0.0, 1.0, float(threshold), 0.01)
time_window_hours = st.sidebar.selectbox("Graph time window", [6, 12, 24, 72, 168], index=2)
max_edges = st.sidebar.slider("Maximum graph edges", min_value=20, max_value=300, value=120)

st.sidebar.divider()
show_model_results = st.sidebar.checkbox("Show model comparison results", value=True)

if uploaded is None:
    st.info("Upload a transaction CSV to start scoring.")
    st.stop()

temp_path.parent.mkdir(parents=True, exist_ok=True)
temp_path.write_bytes(uploaded.getvalue())

df = load_transactions(temp_path)
df = add_basic_transaction_features(df)

needs_graph = any("graph" in str(feature_set) for _ in [0]) or any(
    col in feature_columns
    for col in [
        "sender_n_sent_prev",
        "sender_tx_count_1h_prev",
        "sender_in_degree_prev",
        "sender_n2v_0",
    ]
)

if needs_graph:
    df = add_historical_graph_features(df)
    df = add_rolling_graph_features(df)
    df = add_temporal_centrality_and_motif_features(df)

needs_static_graph = any(
    col in feature_columns
    for col in [
        "sender_n_sent",
        "sender_total_sent",
        "sender_unique_receivers",
        "receiver_n_received",
        "receiver_total_received",
        "receiver_unique_senders",
    ]
)

if needs_static_graph:
    df = add_account_graph_aggregate_features(df)

include_graph_features = any(
    col in feature_columns
    for col in [
        "sender_n_sent_prev",
        "sender_tx_count_1h_prev",
        "sender_in_degree_prev",
        "cycle_proxy_score",
    ]
)

X, y = make_model_matrix(df, include_graph_features=include_graph_features)

if "node2vec_embeddings" in bundle:
    n2v_features = make_node2vec_embedding_features(
        df,
        bundle["node2vec_embeddings"],
        bundle["node2vec_dimensions"],
    )

    X = pd.concat(
        [X.reset_index(drop=True), n2v_features.reset_index(drop=True)],
        axis=1,
    )

df["sender_n2v_known"] = n2v_features["sender_n2v_known"].values
df["receiver_n2v_known"] = n2v_features["receiver_n2v_known"].values

# Align feature columns with training.
X = X.reindex(columns=feature_columns, fill_value=0)

scores = model.predict_proba(X)[:, 1]

result = df.copy()
result["suspicious_score"] = scores
result["predicted_label"] = (result["suspicious_score"] >= score_threshold).astype(int)

st.subheader("Scoring Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total transactions", f"{len(result):,}")
c2.metric("Predicted suspicious", f"{int(result['predicted_label'].sum()):,}")
c3.metric("Highest score", f"{result['suspicious_score'].max():.4f}")
c4.metric("Average score", f"{result['suspicious_score'].mean():.4f}")

st.subheader("Top Suspicious Transactions")

display_cols = [
    c for c in [
        "timestamp",
        "sender_id",
        "receiver_id",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "suspicious_score",
        "predicted_label",
        "is_laundering",
    ] if c in result.columns
]

ranked = result.sort_values("suspicious_score", ascending=False).head(top_n).copy()
st.dataframe(ranked[display_cols], use_container_width=True)

selected_pos = st.selectbox(
    "Select transaction rank for graph visualization",
    options=list(range(len(ranked))),
    format_func=lambda i: (
        f"Rank {i+1} | score={ranked.iloc[i]['suspicious_score']:.4f} | "
        f"{ranked.iloc[i]['sender_id']} → {ranked.iloc[i]['receiver_id']}"
    )
)

selected_idx = ranked.index[selected_pos]
selected_row = result.loc[selected_idx]

st.subheader("Selected Transaction")

st.write({
    "timestamp": str(selected_row.get("timestamp", "")),
    "sender": selected_row.get("sender_id", ""),
    "receiver": selected_row.get("receiver_id", ""),
    "amount_paid": selected_row.get("amount_paid", ""),
    "payment_format": selected_row.get("payment_format", ""),
    "suspicious_score": float(selected_row.get("suspicious_score", 0)),
    "predicted_label": int(selected_row.get("predicted_label", 0)),
})

st.subheader("Graph-Based Explanation Features")

explanation_df = build_explanation_table(selected_row)

if explanation_df.empty:
    st.info("No explanation features available for this transaction.")
else:
    st.dataframe(explanation_df, use_container_width=True)

    st.caption(
        "These features summarize the selected transaction and the surrounding account behavior. "
        "High outgoing/incoming counts, repeated counterparties, cross-bank movement, and high total flow "
        "can indicate suspicious transaction patterns such as fan-in, fan-out, or layering."
    )

st.subheader("Local Transaction Graph")

local_df, selected_info = get_local_transactions(
    result,
    selected_idx=selected_idx,
    time_window_hours=time_window_hours,
    max_edges=max_edges,
)

graph_path = build_pyvis_graph(
    local_df=local_df,
    selected_sender=selected_info["sender"],
    selected_receiver=selected_info["receiver"],
    output_path=str(PROJECT_ROOT / "reports/figures/local_transaction_graph.html"),
)

with open(graph_path, "r", encoding="utf-8") as f:
    html = f.read()

components.html(html, height=700, scrolling=True)

st.subheader("Local Graph Summary")

local_accounts = set(local_df["sender_id"]).union(set(local_df["receiver_id"]))

summary_data = {
    "Local transactions shown": len(local_df),
    "Unique accounts shown": len(local_accounts),
    "Unique senders": local_df["sender_id"].nunique(),
    "Unique receivers": local_df["receiver_id"].nunique(),
}

if "amount_paid" in local_df.columns:
    summary_data["Total amount in local graph"] = float(local_df["amount_paid"].sum())

if "is_laundering" in local_df.columns:
    summary_data["Known laundering labels in local graph"] = int(local_df["is_laundering"].sum())

st.write(summary_data)

st.subheader("Download Results")
csv = result.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download scored transactions",
    data=csv,
    file_name="scored_transactions.csv",
    mime="text/csv",
)


if show_model_results:
    st.subheader("Model Comparison Results")

    comparison_path = PROJECT_ROOT / "reports/tables/four_model_comparison_temporal_graph_final.csv"
    node2vec_metrics_path = PROJECT_ROOT / "reports/tables/lgbm_node2vec_metrics_final.csv"
    topk_path = PROJECT_ROOT / "reports/tables/top_k_evaluation_node2vec_final.csv"

    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        st.write("Four-algorithm comparison on temporal graph feature set")
        st.dataframe(comparison_df, use_container_width=True)

        metric_cols = ["model", "val_pr_auc", "test_pr_auc", "test_f1", "test_precision", "test_recall"]
        available_metric_cols = [c for c in metric_cols if c in comparison_df.columns]

        if available_metric_cols:
            chart_df = comparison_df[available_metric_cols].copy()
            st.bar_chart(chart_df.set_index("model")[["val_pr_auc", "test_f1"]])

    else:
        st.info("Four-model comparison table not found.")

    if node2vec_metrics_path.exists():
        node2vec_df = pd.read_csv(node2vec_metrics_path)
        st.write("Final node2vec-enhanced LightGBM result")
        st.dataframe(node2vec_df, use_container_width=True)

    if topk_path.exists():
        topk_df = pd.read_csv(topk_path)
        st.write("Top-K evaluation for final model")
        st.dataframe(topk_df, use_container_width=True)