from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data.load_data import load_transactions
from src.features.tabular_features import add_basic_transaction_features, make_model_matrix
from src.features.graph_features import add_account_graph_aggregate_features
from src.features.historical_graph_features import add_historical_graph_features
from src.features.rolling_graph_features import add_rolling_graph_features
from src.visualization.graph_viz import get_local_transactions, build_pyvis_graph


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

model_path = Path("models/best_model_bundle.joblib")

if not model_path.exists():
    st.error("No trained model bundle found at models/best_model_bundle.joblib.")
    st.stop()

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

if uploaded is None:
    st.info("Upload a transaction CSV to start scoring.")
    st.stop()

temp_path = Path("data/sample/uploaded_transactions.csv")
temp_path.parent.mkdir(parents=True, exist_ok=True)
temp_path.write_bytes(uploaded.getvalue())

df = load_transactions(temp_path)
df = add_basic_transaction_features(df)

if feature_set == "raw_plus_graph":
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

    needs_historical_graph = any(
        col in feature_columns
        for col in [
            "sender_n_sent_prev",
            "sender_total_sent_prev",
            "receiver_n_received_prev",
            "receiver_total_received_prev",
            "pair_n_prev",
            "pair_total_amount_prev",
            "fan_out_score",
            "fan_in_score",
            "pair_repeat_score",
            "graph_activity_score",
        ]
    )

    needs_rolling_graph = any(
        col in feature_columns
        for col in [
            "sender_tx_count_1h_prev",
            "sender_amount_sum_1h_prev",
            "receiver_tx_count_1h_prev",
            "receiver_amount_sum_1h_prev",
            "sender_tx_count_24h_prev",
            "sender_amount_sum_24h_prev",
            "receiver_tx_count_24h_prev",
            "receiver_amount_sum_24h_prev",
            "sender_time_since_last_tx_hours",
            "receiver_time_since_last_tx_hours",
            "rolling_fan_out_score_1h",
            "rolling_fan_in_score_1h",
            "rolling_fan_out_score_24h",
            "rolling_fan_in_score_24h",
        ]
    )

    if needs_static_graph:
        df = add_account_graph_aggregate_features(df)

    if needs_historical_graph:
        df = add_historical_graph_features(df)

    if needs_rolling_graph:
        df = add_rolling_graph_features(df)

X, y = make_model_matrix(df, include_graph_features=(feature_set == "raw_plus_graph"))

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
    output_path="reports/figures/local_transaction_graph.html",
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