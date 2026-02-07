"""Exploratory data analysis for LOBSTER data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_plot(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_eda(df: pd.DataFrame, out_dir: Path, ticker: str, date: str) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    time_sec = df["time_ns"] / 1e9

    fig, ax = plt.subplots()
    ax.plot(time_sec, df["mid"], linewidth=0.8)
    ax.set_title("Midprice over time")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Midprice")
    _save_plot(fig, out_dir / f"midprice_{ticker}_{date}.png")

    fig, ax = plt.subplots()
    ax.plot(time_sec, df["spread"], linewidth=0.8)
    ax.set_title("Spread over time")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Spread")
    _save_plot(fig, out_dir / f"spread_{ticker}_{date}.png")

    fig, ax = plt.subplots()
    ax.hist(df["spread"].dropna(), bins=50)
    ax.set_title("Spread distribution")
    ax.set_xlabel("Spread")
    ax.set_ylabel("Count")
    _save_plot(fig, out_dir / f"spread_dist_{ticker}_{date}.png")

    fig, ax = plt.subplots()
    event_counts = df["event_type"].value_counts().sort_index()
    ax.bar(event_counts.index.astype(str), event_counts.values)
    ax.set_title("Event type distribution")
    ax.set_xlabel("Event type")
    ax.set_ylabel("Count")
    _save_plot(fig, out_dir / f"event_types_{ticker}_{date}.png")

    fig, ax = plt.subplots()
    seconds = (df["time_ns"] // 1_000_000_000).astype(int)
    events_per_sec = seconds.value_counts().sort_index()
    ax.plot(events_per_sec.index, events_per_sec.values)
    ax.set_title("Events per second")
    ax.set_xlabel("Second")
    ax.set_ylabel("Events")
    _save_plot(fig, out_dir / f"events_per_sec_{ticker}_{date}.png")

    fig, ax = plt.subplots()
    bins = pd.qcut(df["topk_imbalance"].fillna(0.0), q=10, duplicates="drop")
    grouped = df.groupby(bins)["raw_move_H1"].mean()
    ax.plot(grouped.index.astype(str), grouped.values, marker="o")
    ax.set_title("Imbalance vs future return")
    ax.set_xlabel("Imbalance decile")
    ax.set_ylabel("Mean future move (H1)")
    ax.tick_params(axis="x", rotation=45)
    _save_plot(fig, out_dir / f"imbalance_future_{ticker}_{date}.png")

    fig, ax = plt.subplots()
    ax.scatter(df["realized_volatility"], df["spread"], s=3, alpha=0.5)
    ax.set_title("Volatility vs spread")
    ax.set_xlabel("Realized volatility")
    ax.set_ylabel("Spread")
    _save_plot(fig, out_dir / f"vol_vs_spread_{ticker}_{date}.png")

    fig, ax = plt.subplots()
    is_cancel = df["event_type"].isin([2, 3]).astype(int)
    is_submit = (df["event_type"] == 1).astype(int)
    ratio = (
        is_cancel.groupby(seconds).sum() / is_submit.groupby(seconds).sum().replace(0, np.nan)
    )
    ax.plot(ratio.index, ratio.values)
    ax.set_title("Cancellation/submission ratio")
    ax.set_xlabel("Second")
    ax.set_ylabel("Cancel/Submit")
    _save_plot(fig, out_dir / f"cancel_submit_{ticker}_{date}.png")

    spread_q = df["spread"].quantile([0.33, 0.66])
    vol_q = df["realized_volatility"].quantile([0.33, 0.66])

    spread_regime = pd.cut(
        df["spread"],
        bins=[-np.inf, spread_q.iloc[0], spread_q.iloc[1], np.inf],
        labels=["tight", "normal", "wide"],
    )
    vol_regime = pd.cut(
        df["realized_volatility"],
        bins=[-np.inf, vol_q.iloc[0], vol_q.iloc[1], np.inf],
        labels=["low", "normal", "high"],
    )
    regime_counts = df.groupby([spread_regime, vol_regime]).size().rename("count")

    stats["rows"] = float(len(df))
    stats["time_start"] = float(time_sec.min())
    stats["time_end"] = float(time_sec.max())
    stats["avg_spread"] = float(df["spread"].mean())
    stats["avg_mid"] = float(df["mid"].mean())
    stats["regime_counts"] = regime_counts.to_dict()
    return stats
