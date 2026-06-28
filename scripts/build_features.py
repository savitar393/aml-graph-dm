from __future__ import annotations

import argparse
from pathlib import Path

from src.data.load_data import load_transactions, save_parquet
from src.features.tabular_features import add_basic_transaction_features
from src.features.graph_features import add_account_graph_aggregate_features
from src.features.historical_graph_features import add_historical_graph_features
from src.features.rolling_graph_features import add_rolling_graph_features
from src.features.advanced_graph_features import add_temporal_centrality_and_motif_features
from src.features.snapshot_graph_features import add_snapshot_pagerank_community_features


def output_name(dataset: str) -> str:
    return dataset.lower().replace("-", "_") + "_features.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HI-Small")
    parser.add_argument("--nrows", type=int, default=500_000)
    parser.add_argument("--use-static-graph", action="store_true")
    parser.add_argument("--use-historical-graph", action="store_true")
    parser.add_argument("--use-rolling-graph", action="store_true")
    parser.add_argument("--use-advanced-graph", action="store_true")
    parser.add_argument("--use-snapshot-graph", action="store_true")
    args = parser.parse_args()

    raw_path = Path(f"data/raw/{args.dataset}_Trans.csv")
    out_path = Path("data/processed") / output_name(args.dataset)

    print("Loading:", raw_path)
    print("Rows:", args.nrows)

    df = load_transactions(raw_path, nrows=args.nrows)
    df = add_basic_transaction_features(df)

    if args.use_static_graph:
        print("Adding static aggregate graph features...")
        df = add_account_graph_aggregate_features(df)

    if args.use_historical_graph:
        print("Adding historical leakage-safe graph features...")
        df = add_historical_graph_features(df)

    if args.use_rolling_graph:
        print("Adding rolling-window graph features...")
        df = add_rolling_graph_features(df)

    if args.use_advanced_graph:
        print("Adding temporal centrality and motif/cycle-proxy features...")
        df = add_temporal_centrality_and_motif_features(df)

    if args.use_snapshot_graph:
        print("Adding PageRank and community snapshot features...")
        df = add_snapshot_pagerank_community_features(df)
    save_parquet(df, out_path)

    print("Saved:", out_path)
    print("Shape:", df.shape)
    print("Label distribution:")
    print(df["is_laundering"].value_counts())


if __name__ == "__main__":
    main()