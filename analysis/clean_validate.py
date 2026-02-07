"""Sanity checks and validation for LOBSTER data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

DUMMY_BID = -9999999999
DUMMY_ASK = 9999999999


def _level_columns(prefix: str, num_levels: int) -> List[str]:
    return [f"{prefix}_{level}" for level in range(1, num_levels + 1)]


def add_halt_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_halt"] = (df["event_type"] == 7).astype("int8")
    return df


def sanity_checks(df: pd.DataFrame, num_levels: int) -> Dict[str, int]:
    best_bid = df["bid_price_1"].replace(DUMMY_BID, np.nan)
    best_ask = df["ask_price_1"].replace(DUMMY_ASK, np.nan)
    spread_ok = (best_bid < best_ask) | best_bid.isna() | best_ask.isna()

    bid_sizes = df[_level_columns("bid_size", num_levels)]
    ask_sizes = df[_level_columns("ask_size", num_levels)]
    sizes_ok = (bid_sizes >= 0).all(axis=None) and (ask_sizes >= 0).all(axis=None)

    bid_prices = df[_level_columns("bid_price", num_levels)].replace(DUMMY_BID, np.nan)
    ask_prices = df[_level_columns("ask_price", num_levels)].replace(DUMMY_ASK, np.nan)

    bid_monotonic = (
        np.nan_to_num(bid_prices.values, nan=-np.inf)[:, :-1]
        >= np.nan_to_num(bid_prices.values, nan=-np.inf)[:, 1:]
    ).all()
    ask_monotonic = (
        np.nan_to_num(ask_prices.values, nan=np.inf)[:, :-1]
        <= np.nan_to_num(ask_prices.values, nan=np.inf)[:, 1:]
    ).all()

    return {
        "rows": int(len(df)),
        "spread_violations": int((~spread_ok).sum()),
        "sizes_non_negative": int(sizes_ok),
        "bid_monotonic": int(bid_monotonic),
        "ask_monotonic": int(ask_monotonic),
        "halt_rows": int((df["event_type"] == 7).sum()),
    }


def write_sanity(summary: Dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
