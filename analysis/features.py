"""Microstructure feature engineering."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .clean_validate import DUMMY_ASK, DUMMY_BID


def _level_arrays(df: pd.DataFrame, prefix: str, num_levels: int) -> np.ndarray:
    cols = [f"{prefix}_{level}" for level in range(1, num_levels + 1)]
    return df[cols].to_numpy()


def _mask_dummy_prices(prices: np.ndarray, side: str) -> np.ndarray:
    if side == "bid":
        return np.where(prices == DUMMY_BID, np.nan, prices)
    return np.where(prices == DUMMY_ASK, np.nan, prices)


def _topk_depth(sizes: np.ndarray, k: int) -> np.ndarray:
    return np.nansum(sizes[:, :k], axis=1)


def _rolling_event_rate(time_ns: np.ndarray, window_ns: int = int(1e9)) -> np.ndarray:
    rates = np.zeros_like(time_ns, dtype="float64")
    left = 0
    for idx, t in enumerate(time_ns):
        while time_ns[left] < t - window_ns:
            left += 1
        window_count = idx - left + 1
        rates[idx] = window_count / (window_ns / 1e9)
    return rates


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).sum()


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=2).std().fillna(0.0)


def compute_features(df: pd.DataFrame, num_levels: int, topk: int) -> pd.DataFrame:
    bid_prices = _mask_dummy_prices(_level_arrays(df, "bid_price", num_levels), "bid")
    ask_prices = _mask_dummy_prices(_level_arrays(df, "ask_price", num_levels), "ask")
    bid_sizes = _level_arrays(df, "bid_size", num_levels)
    ask_sizes = _level_arrays(df, "ask_size", num_levels)

    best_bid = bid_prices[:, 0]
    best_ask = ask_prices[:, 0]
    mid = (best_bid + best_ask) / 2.0
    spread = best_ask - best_bid

    level1_imb = (bid_sizes[:, 0] - ask_sizes[:, 0]) / (
        bid_sizes[:, 0] + ask_sizes[:, 0]
    )
    bid_depth_k = _topk_depth(bid_sizes, topk)
    ask_depth_k = _topk_depth(ask_sizes, topk)
    topk_imb = (bid_depth_k - ask_depth_k) / (bid_depth_k + ask_depth_k)

    microprice = (
        best_bid * ask_sizes[:, 0] + best_ask * bid_sizes[:, 0]
    ) / (bid_sizes[:, 0] + ask_sizes[:, 0])

    time_ns = df["time_ns"].to_numpy()
    event_rate = _rolling_event_rate(time_ns)

    sign = np.select(
        [df["event_type"].isin([1]), df["event_type"].isin([2, 3, 4, 5])],
        [1, -1],
        default=0,
    )
    signed_flow = df["direction"] * df["size"] * sign
    order_flow_imb = _rolling_sum(signed_flow, window=100)

    mid_series = pd.Series(mid)
    mid_ret = mid_series.pct_change().fillna(0.0)
    realized_vol = _rolling_std(mid_ret, window=100)

    features = df.copy()
    features["best_bid"] = best_bid
    features["best_ask"] = best_ask
    features["mid"] = mid
    features["spread"] = spread
    features["level1_imbalance"] = level1_imb
    features["topk_imbalance"] = topk_imb
    features["depth_topk_bid"] = bid_depth_k
    features["depth_topk_ask"] = ask_depth_k
    features["depth_total_topk"] = bid_depth_k + ask_depth_k
    features["microprice"] = microprice
    features["event_rate"] = event_rate
    features["order_flow_imbalance"] = order_flow_imb
    features["realized_volatility"] = realized_vol
    return features
