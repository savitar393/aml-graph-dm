from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data.load_data import load_transactions
from src.features.tabular_features import add_basic_transaction_features, make_model_matrix
from src.features.graph_features import add_account_graph_aggregate_features


st.set_page_config(page_title="AML Graph Scoring Demo", layout="wide")

st.title("AML Graph-Enhanced Scoring Demo")
st.write(
    "Upload a transaction CSV with the same schema as the training data. "
    "The app will build basic transaction and graph aggregate features, then score suspicious transactions."
)

model_path = Path("models/best_model.joblib")
if not model_path.exists():
    st.warning("No trained model found at models/best_model.joblib. Train a model first.")
else:
    model = joblib.load(model_path)

uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded is not None:
    temp_path = Path("data/sample/uploaded_transactions.csv")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(uploaded.getvalue())

    df = load_transactions(temp_path)
    df = add_basic_transaction_features(df)
    df = add_account_graph_aggregate_features(df)
    X, _ = make_model_matrix(df, include_graph_features=True)

    if model_path.exists():
        # Align columns if model was saved with feature_names_in_
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            X = X[list(model.feature_names_in_)]

        scores = model.predict_proba(X)[:, 1]
        result = df.copy()
        result["suspicious_score"] = scores
        result["predicted_label"] = (result["suspicious_score"] >= 0.5).astype(int)

        st.subheader("Top suspicious transactions")
        display_cols = [
            c for c in [
                "timestamp", "sender_id", "receiver_id", "amount_paid",
                "payment_currency", "payment_format", "suspicious_score", "predicted_label"
            ] if c in result.columns
        ]
        st.dataframe(result.sort_values("suspicious_score", ascending=False)[display_cols].head(100))

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored transactions", csv, "scored_transactions.csv", "text/csv")
