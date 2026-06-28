from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HI-Small")
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    raw_path = Path(f"data/raw/{args.dataset}_Trans.csv")
    out_path = Path(f"reports/tables/{args.dataset.lower().replace('-', '_')}_label_timeline.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    start_row = 0

    for chunk_id, chunk in enumerate(pd.read_csv(raw_path, chunksize=args.chunksize), start=1):
        n = len(chunk)
        label_col = "Is Laundering"

        if "Timestamp" in chunk.columns:
            ts = pd.to_datetime(chunk["Timestamp"], errors="coerce")
            min_ts = ts.min()
            max_ts = ts.max()
        else:
            min_ts = None
            max_ts = None

        laundering_count = int(chunk[label_col].sum())
        normal_count = int(n - laundering_count)

        rows.append({
            "chunk_id": chunk_id,
            "start_row": start_row,
            "end_row": start_row + n - 1,
            "rows": n,
            "normal_count": normal_count,
            "laundering_count": laundering_count,
            "laundering_rate": laundering_count / n,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
        })

        print(
            f"chunk={chunk_id} rows={n:,} "
            f"laundering={laundering_count:,} "
            f"rate={laundering_count / n:.6f}"
        )

        start_row += n

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)

    print("\nSaved:", out_path)
    print("\nTotal:")
    print(result[["rows", "normal_count", "laundering_count"]].sum())
    print("\nTimeline:")
    print(result)


if __name__ == "__main__":
    main()