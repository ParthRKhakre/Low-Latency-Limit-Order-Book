"""Configuration loader for LOBSTER analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class Config:
    num_levels: int = 10
    topk: int = 5
    horizons_events: List[int] = None
    export_format: str = "parquet"
    chunk_rows: int = 500000
    output_dir: str = "results"
    plotting: bool = True

    @staticmethod
    def load(path: Path) -> "Config":
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        horizons = data.get("horizons_events", [1, 5, 10])
        return Config(
            num_levels=int(data.get("num_levels", 10)),
            topk=int(data.get("topk", 5)),
            horizons_events=[int(h) for h in horizons],
            export_format=str(data.get("export_format", "parquet")),
            chunk_rows=int(data.get("chunk_rows", 500000)),
            output_dir=str(data.get("output_dir", "results")),
            plotting=bool(data.get("plotting", True)),
        )
