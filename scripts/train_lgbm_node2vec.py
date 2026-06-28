from __future__ import annotations

from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
import networkx as nx

from lightgbm import LGBMClassifier
from node2vec import Node2Vec

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.data.load_data import chronological_split
from src.features.tabular_features import make_model_matrix
from src.evaluation.thresholds import find_best_threshold, top_k_table


def sample_training_data(X, y, neg_per_pos: int = 100, random_state: int = 42):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_pos = len(pos_idx)
    if n_pos == 0:
        raise ValueError("No positive laundering samples in training split.")

    n_neg = min(len(neg_idx), n_pos * neg_per_pos)
    sampled_neg_idx = neg_idx.to_series().sample(n=n_neg, random_state=random_state).index
    sampled_idx = pos_idx.union(sampled_neg_idx)

    return X.loc[sampled_idx], y.loc[sampled_idx]


def build_train_graph(
    train_df: pd.DataFrame,
    max_edges: int = 300_000,
    weight_mode: str = "tx_count",
) -> nx.DiGraph:
    """Build node2vec graph from training rows only."""
    edges = (
        train_df
        .groupby(["sender_id", "receiver_id"])
        .agg(
            tx_count=("amount_paid", "size"),
            total_amount=("amount_paid", "sum"),
        )
        .reset_index()
    )

    if weight_mode == "amount":
        edges["weight"] = np.log1p(edges["total_amount"].astype(float))
    else:
        edges["weight"] = edges["tx_count"].astype(float)

    # Bound graph size for runtime. Keep strongest transaction relationships.
    if len(edges) > max_edges:
        edges = edges.sort_values(["tx_count", "total_amount"], ascending=False).head(max_edges)

    G = nx.DiGraph()

    for row in edges.itertuples(index=False):
        G.add_edge(
            str(row.sender_id),
            str(row.receiver_id),
            weight=float(row.weight),
        )

    return G


def train_node2vec_embeddings(
    G: nx.DiGraph,
    dimensions: int = 16,
    walk_length: int = 10,
    num_walks: int = 5,
    workers: int = 4,
) -> dict[str, np.ndarray]:
    """Train node2vec on training graph only."""
    if G.number_of_nodes() == 0:
        raise ValueError("Training graph has no nodes.")

    print(f"Node2Vec graph nodes: {G.number_of_nodes():,}")
    print(f"Node2Vec graph edges: {G.number_of_edges():,}")

    n2v = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        weight_key="weight",
        quiet=False,
    )

    w2v = n2v.fit(
        window=5,
        min_count=1,
        batch_words=2048,
    )

    embeddings = {
        str(node): w2v.wv[str(node)].astype(np.float32)
        for node in w2v.wv.index_to_key
    }

    return embeddings


def make_embedding_features(
    df: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    dimensions: int,
) -> pd.DataFrame:
    """Create sender and receiver embedding columns for transaction-level modeling."""
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


def evaluate_scores(y_true, scores, threshold: float = 0.5) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "alert_count": int(pred.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/processed/hi_small_features.parquet")
    parser.add_argument("--neg-per-pos", type=int, default=100)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--walk-length", type=int, default=10)
    parser.add_argument("--num-walks", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=300_000)
    parser.add_argument("--weight-mode", choices=["tx_count", "amount"], default="tx_count")
    args = parser.parse_args()

    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    print("Loading:", args.data_path)
    df = pd.read_parquet(args.data_path)

    train_df, val_df, test_df = chronological_split(df, train_size=0.60, val_size=0.20)

    print("\nSplit label counts:")
    print("Train:")
    print(train_df["is_laundering"].value_counts())
    print("Validation:")
    print(val_df["is_laundering"].value_counts())
    print("Test:")
    print(test_df["is_laundering"].value_counts())

    print("\nBuilding train-only graph for node2vec...")
    G = build_train_graph(
        train_df,
        max_edges=args.max_edges,
        weight_mode=args.weight_mode,
    )

    print("\nTraining node2vec embeddings...")
    embeddings = train_node2vec_embeddings(
        G,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        workers=args.workers,
    )

    print("\nBuilding base model matrices...")
    X_train_base, y_train = make_model_matrix(train_df, include_graph_features=True)
    X_val_base, y_val = make_model_matrix(val_df, include_graph_features=True)
    X_test_base, y_test = make_model_matrix(test_df, include_graph_features=True)

    X_val_base = X_val_base.reindex(columns=X_train_base.columns, fill_value=0)
    X_test_base = X_test_base.reindex(columns=X_train_base.columns, fill_value=0)

    print("\nAttaching sender/receiver node2vec embeddings...")
    train_emb = make_embedding_features(train_df, embeddings, args.dimensions)
    val_emb = make_embedding_features(val_df, embeddings, args.dimensions)
    test_emb = make_embedding_features(test_df, embeddings, args.dimensions)

    coverage = pd.DataFrame([
        {
            "split": "train",
            "sender_known_rate": float(train_emb["sender_n2v_known"].mean()),
            "receiver_known_rate": float(train_emb["receiver_n2v_known"].mean()),
        },
        {
            "split": "validation",
            "sender_known_rate": float(val_emb["sender_n2v_known"].mean()),
            "receiver_known_rate": float(val_emb["receiver_n2v_known"].mean()),
        },
        {
            "split": "test",
            "sender_known_rate": float(test_emb["sender_n2v_known"].mean()),
            "receiver_known_rate": float(test_emb["receiver_n2v_known"].mean()),
        },
    ])
    coverage.to_csv(out_dir / "node2vec_embedding_coverage.csv", index=False)
    print("\nNode2Vec embedding coverage:")
    print(coverage)

    X_train = pd.concat([X_train_base.reset_index(drop=True), train_emb.reset_index(drop=True)], axis=1)
    X_val = pd.concat([X_val_base.reset_index(drop=True), val_emb.reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test_base.reset_index(drop=True), test_emb.reset_index(drop=True)], axis=1)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_sample, y_sample = sample_training_data(
        X_train,
        y_train,
        neg_per_pos=args.neg_per_pos,
        random_state=42,
    )

    pos = max(int(y_sample.sum()), 1)
    neg = max(len(y_sample) - pos, 1)
    scale_pos_weight = neg / pos

    print("\nSampled training data:")
    print("Shape:", X_sample.shape)
    print("Positive:", pos)
    print("Negative:", neg)
    print("scale_pos_weight:", scale_pos_weight)

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining LightGBM + node2vec...")
    model.fit(X_sample, y_sample)

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_scores(y_val, val_scores, threshold=0.5)
    test_metrics = evaluate_scores(y_test, test_scores, threshold=0.5)

    metrics = pd.DataFrame([
        {
            "model": "lightgbm_node2vec",
            "feature_set": "raw_plus_advanced_graph_plus_node2vec",
            "neg_per_pos": args.neg_per_pos,
            "node2vec_dimensions": args.dimensions,
            "node2vec_walk_length": args.walk_length,
            "node2vec_num_walks": args.num_walks,
            "node2vec_max_edges": args.max_edges,
            "feature_count": X_train.shape[1],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_tp": test_metrics["tp"],
            "test_fp": test_metrics["fp"],
            "test_tn": test_metrics["tn"],
            "test_fn": test_metrics["fn"],
            "test_alert_count": test_metrics["alert_count"],
        }
    ])

    metrics.to_csv(out_dir / "lgbm_node2vec_metrics.csv", index=False)

    print("\nNode2Vec model metrics:")
    print(metrics)

    threshold_result = find_best_threshold(y_val, val_scores, metric="f1")
    best_threshold = threshold_result["best_threshold"]

    default_pred = (test_scores >= 0.5).astype(int)
    tuned_pred = (test_scores >= best_threshold).astype(int)

    final_eval = pd.DataFrame([
        {
            "setting": "default_threshold_0.5",
            "threshold": 0.5,
            "pr_auc": average_precision_score(y_test, test_scores),
            "f1": f1_score(y_test, default_pred, zero_division=0),
            "precision": precision_score(y_test, default_pred, zero_division=0),
            "recall": recall_score(y_test, default_pred, zero_division=0),
            "alert_count": int(default_pred.sum()),
        },
        {
            "setting": "tuned_threshold_validation",
            "threshold": best_threshold,
            "pr_auc": average_precision_score(y_test, test_scores),
            "f1": f1_score(y_test, tuned_pred, zero_division=0),
            "precision": precision_score(y_test, tuned_pred, zero_division=0),
            "recall": recall_score(y_test, tuned_pred, zero_division=0),
            "alert_count": int(tuned_pred.sum()),
        },
    ])

    cm = confusion_matrix(y_test, tuned_pred, labels=[0, 1])
    confusion_df = pd.DataFrame(
        cm,
        index=["actual_normal", "actual_laundering"],
        columns=["predicted_normal", "predicted_laundering"],
    )

    topk = top_k_table(y_test, test_scores, k_values=(50, 100, 200, 500, 1000))

    final_eval.to_csv(out_dir / "final_threshold_evaluation_node2vec.csv", index=False)
    confusion_df.to_csv(out_dir / "confusion_matrix_tuned_node2vec.csv")
    topk.to_csv(out_dir / "top_k_evaluation_node2vec.csv", index=False)
    threshold_result["threshold_table"].to_csv(out_dir / "threshold_search_table_node2vec.csv", index=False)

    joblib.dump(
        {
            "model": model,
            "model_name": "lightgbm_node2vec",
            "feature_set": "raw_plus_graph_node2vec",
            "feature_columns": list(X_train.columns),
            "threshold": float(best_threshold),
            "node2vec_embeddings": embeddings,
            "node2vec_dimensions": args.dimensions,
            "node2vec_default_vector": [0.0] * args.dimensions,
            "training_note": (
                "Node2Vec embeddings trained on chronological training graph only. "
                "Validation/test graph edges were not used during embedding training."
            ),
        },
        "models/node2vec_model_bundle.joblib",
    )

    print("\nFinal threshold evaluation:")
    print(final_eval)

    print("\nTop-K evaluation:")
    print(topk)

    print("\nTuned confusion matrix:")
    print(confusion_df)

    print("\nSaved:")
    print(out_dir / "lgbm_node2vec_metrics.csv")
    print(out_dir / "node2vec_embedding_coverage.csv")
    print(out_dir / "final_threshold_evaluation_node2vec.csv")
    print(out_dir / "top_k_evaluation_node2vec.csv")
    print(out_dir / "confusion_matrix_tuned_node2vec.csv")
    print("models/node2vec_model_bundle.joblib")


if __name__ == "__main__":
    main()