from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_window_name(window: str) -> str:
    return (
        window.lower()
        .replace(" ", "")
        .replace("h", "h")
        .replace("d", "d")
    )


def _add_group_rolling_amount_features(
    out: pd.DataFrame,
    group_col: str,
    prefix: str,
    windows: tuple[str, ...],
    time_col: str = "timestamp",
    amount_col: str = "amount_paid",
) -> pd.DataFrame:
    """Add past-only rolling count/sum/mean/max features for one group column."""
    work = out[[group_col, time_col, amount_col]].copy()
    work["_row_id"] = out.index

    work = work.sort_values([group_col, time_col])

    for window in windows:
        wname = _safe_window_name(window)

        rolled = (
            work.groupby(group_col)
            .rolling(window=window, on=time_col, closed="left")[amount_col]
            .agg(["count", "sum", "mean", "max"])
            .reset_index(drop=True)
        )

        feature_map = {
            "count": f"{prefix}_tx_count_{wname}_prev",
            "sum": f"{prefix}_amount_sum_{wname}_prev",
            "mean": f"{prefix}_amount_mean_{wname}_prev",
            "max": f"{prefix}_amount_max_{wname}_prev",
        }

        for stat, col in feature_map.items():
            values = rolled[stat].fillna(0.0).to_numpy()
            out.loc[work["_row_id"].to_numpy(), col] = values

    return out


def _add_time_since_last_features(
    out: pd.DataFrame,
    group_col: str,
    output_col: str,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Add time since previous transaction for a given group."""
    work = out[[group_col, time_col]].copy()
    work["_row_id"] = out.index
    work = work.sort_values([group_col, time_col])

    prev_time = work.groupby(group_col)[time_col].shift(1)
    delta_hours = (work[time_col] - prev_time).dt.total_seconds() / 3600.0

    out.loc[work["_row_id"].to_numpy(), output_col] = (
        delta_hours.replace([np.inf, -np.inf], np.nan).fillna(-1.0).to_numpy()
    )

    return out


def add_rolling_graph_features(
    df: pd.DataFrame,
    windows: tuple[str, ...] = ("1h", "24h"),
) -> pd.DataFrame:
    """
    Add past-only rolling temporal graph features.

    These features capture short-term sender/receiver bursts while avoiding
    future-information leakage by using closed='left'.
    """
    out = df.copy()

    if "timestamp" not in out.columns:
        raise ValueError("timestamp column is required for rolling graph features.")

    out = out.sort_values("timestamp").reset_index(drop=True)
    out["amount_paid"] = pd.to_numeric(out["amount_paid"], errors="coerce").fillna(0.0)

    out = _add_group_rolling_amount_features(
        out,
        group_col="sender_id",
        prefix="sender",
        windows=windows,
    )

    out = _add_group_rolling_amount_features(
        out,
        group_col="receiver_id",
        prefix="receiver",
        windows=windows,
    )

    out = _add_time_since_last_features(
        out,
        group_col="sender_id",
        output_col="sender_time_since_last_tx_hours",
    )

    out = _add_time_since_last_features(
        out,
        group_col="receiver_id",
        output_col="receiver_time_since_last_tx_hours",
    )

    eps = 1e-9

    for window in windows:
        wname = _safe_window_name(window)

        out[f"rolling_fan_out_score_{wname}"] = (
            np.log1p(out[f"sender_tx_count_{wname}_prev"])
            * np.log1p(out[f"sender_amount_sum_{wname}_prev"])
        )

        out[f"rolling_fan_in_score_{wname}"] = (
            np.log1p(out[f"receiver_tx_count_{wname}_prev"])
            * np.log1p(out[f"receiver_amount_sum_{wname}_prev"])
        )

        out[f"amount_vs_sender_mean_{wname}"] = (
            out["amount_paid"] / (out[f"sender_amount_mean_{wname}_prev"] + eps)
        )

        out[f"amount_vs_receiver_mean_{wname}"] = (
            out["amount_paid"] / (out[f"receiver_amount_mean_{wname}_prev"] + eps)
        )

    return out