"""Load LOBSTER message and orderbook CSVs with alignment."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

MESSAGE_COLUMNS = ["time_sec", "event_type", "order_id", "size", "price", "direction"]


MESSAGE_DTYPES = {
    "time_sec": "float64",
    "event_type": "int8",
    "order_id": "int64",
    "size": "int64",
    "price": "int64",
    "direction": "int8",
}


def _orderbook_columns(num_levels: int) -> list[str]:
    cols = []
    for level in range(1, num_levels + 1):
        cols.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return cols


def _orderbook_dtypes(num_levels: int) -> dict[str, str]:
    dtypes: dict[str, str] = {}
    for level in range(1, num_levels + 1):
        dtypes[f"ask_price_{level}"] = "int64"
        dtypes[f"ask_size_{level}"] = "int64"
        dtypes[f"bid_price_{level}"] = "int64"
        dtypes[f"bid_size_{level}"] = "int64"
    return dtypes


def load_full(message_path: Path, orderbook_path: Path, num_levels: int) -> pd.DataFrame:
    message = pd.read_csv(
        message_path,
        header=None,
        names=MESSAGE_COLUMNS,
        dtype=MESSAGE_DTYPES,
    )
    orderbook = pd.read_csv(
        orderbook_path,
        header=None,
        names=_orderbook_columns(num_levels),
        dtype=_orderbook_dtypes(num_levels),
    )
    if len(message) != len(orderbook):
        raise ValueError("Message and orderbook rows are misaligned.")
    df = pd.concat([message, orderbook], axis=1)
    df["time_ns"] = np.round(df["time_sec"] * 1e9).astype("int64")
    return df


def iter_load(
    message_path: Path,
    orderbook_path: Path,
    num_levels: int,
    chunk_rows: int,
) -> Iterator[pd.DataFrame]:
    message_iter = pd.read_csv(
        message_path,
        header=None,
        names=MESSAGE_COLUMNS,
        dtype=MESSAGE_DTYPES,
        chunksize=chunk_rows,
    )
    orderbook_iter = pd.read_csv(
        orderbook_path,
        header=None,
        names=_orderbook_columns(num_levels),
        dtype=_orderbook_dtypes(num_levels),
        chunksize=chunk_rows,
    )
    for message_chunk, orderbook_chunk in zip(message_iter, orderbook_iter):
        if len(message_chunk) != len(orderbook_chunk):
            raise ValueError("Chunk size mismatch between message and orderbook.")
        df = pd.concat([message_chunk, orderbook_chunk], axis=1)
        df["time_ns"] = np.round(df["time_sec"] * 1e9).astype("int64")
        yield df


def load_lobster(
    message_path: Path,
    orderbook_path: Path,
    num_levels: int,
    chunk_rows: int | None = None,
) -> Tuple[pd.DataFrame | None, Iterator[pd.DataFrame] | None]:
    if chunk_rows:
        return None, iter_load(message_path, orderbook_path, num_levels, chunk_rows)
    return load_full(message_path, orderbook_path, num_levels), None
