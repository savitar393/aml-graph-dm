from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


COLUMN_MAP = {
    "timestamp": "timestamp",
    "time": "timestamp",
    "date": "timestamp",

    "from bank": "from_bank",
    "from_bank": "from_bank",
    "source bank": "from_bank",

    "to bank": "to_bank",
    "to_bank": "to_bank",
    "destination bank": "to_bank",

    "account": "sender_account",
    "account.1": "receiver_account",
    "from account": "sender_account",
    "sender account": "sender_account",
    "sender": "sender_account",
    "source account": "sender_account",
    "to account": "receiver_account",
    "receiver account": "receiver_account",
    "receiver": "receiver_account",
    "destination account": "receiver_account",

    "amount received": "amount_received",
    "received amount": "amount_received",
    "amount paid": "amount_paid",
    "paid amount": "amount_paid",
    "amount": "amount_paid",

    "receiving currency": "receiving_currency",
    "payment currency": "payment_currency",
    "currency": "payment_currency",

    "payment format": "payment_format",
    "payment type": "payment_format",
    "type": "payment_format",

    "is laundering": "is_laundering",
    "is_laundering": "is_laundering",
    "label": "is_laundering",
    "target": "is_laundering",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common IBM AMLworld column names to snake_case names."""
    original_cols = list(df.columns)
    normalized = {}

    # Special handling for duplicate-like IBM columns: Account and Account.1
    for col in original_cols:
        key = str(col).strip().lower()
        normalized[col] = COLUMN_MAP.get(key, key.replace(" ", "_").replace("-", "_"))

    df = df.rename(columns=normalized)

    # If pandas renamed duplicate account columns differently, try to infer.
    if "sender_account" not in df.columns:
        candidates = [c for c in df.columns if "account" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "sender_account"})
            if len(candidates) > 1:
                df = df.rename(columns={candidates[1]: "receiver_account"})

    return df


def load_transactions(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load transactions and create sender_id / receiver_id.

    Parameters
    ----------
    path:
        CSV file path. Example: data/raw/HI-Small_Trans.csv
    nrows:
        Optional number of rows for quick development.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, nrows=nrows)
    df = _normalize_columns(df)

    required = ["sender_account", "receiver_account", "is_laundering"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required normalized columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    if "from_bank" in df.columns:
        df["sender_id"] = df["from_bank"].astype(str) + "_" + df["sender_account"].astype(str)
    else:
        df["sender_id"] = df["sender_account"].astype(str)

    if "to_bank" in df.columns:
        df["receiver_id"] = df["to_bank"].astype(str) + "_" + df["receiver_account"].astype(str)
    else:
        df["receiver_id"] = df["receiver_account"].astype(str)

    df["is_laundering"] = df["is_laundering"].astype(int)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_size: float = 0.60,
    val_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split in chronological order to reduce temporal leakage."""
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
