"""Label generation for mid-price movement prediction."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_labels(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    labeled = df.copy()
    for horizon in horizons:
        future_mid = labeled["mid"].shift(-horizon)
        raw_move = future_mid - labeled["mid"]
        threshold = np.maximum(labeled["spread"] / 2.0, 1)
        label = np.where(raw_move > threshold, 1, np.where(raw_move < -threshold, -1, 0))
        labeled[f"raw_move_H{horizon}"] = raw_move
        labeled[f"label_H{horizon}"] = label.astype("int8")
    return labeled
