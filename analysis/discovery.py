"""Dataset discovery utilities."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

PAIR = Tuple[Path, Path, str, str]

MESSAGE_RE = re.compile(r"(?P<ticker>[A-Za-z]+)_(?P<date>\d{4}-\d{2}-\d{2}).*message.*\.csv$")
ORDERBOOK_RE = re.compile(r"(?P<ticker>[A-Za-z]+)_(?P<date>\d{4}-\d{2}-\d{2}).*orderbook.*\.csv$")


def _infer_from_filename(path: Path) -> Tuple[str, str] | None:
    match = MESSAGE_RE.match(path.name) or ORDERBOOK_RE.match(path.name)
    if not match:
        return None
    return match.group("ticker"), match.group("date")


def _pair_from_directory(directory: Path) -> Tuple[Path, Path] | None:
    message = directory / "message.csv"
    orderbook = directory / "orderbook.csv"
    if message.exists() and orderbook.exists():
        return message, orderbook
    return None


def discover_pairs(data_root: Path, repo_root: Path) -> List[PAIR]:
    pairs: Dict[Tuple[str, str], Tuple[Path, Path]] = {}
    if data_root.exists():
        for subdir in data_root.rglob("*"):
            if not subdir.is_dir():
                continue
            found = _pair_from_directory(subdir)
            if found:
                ticker = subdir.parent.name
                date = subdir.name
                pairs[(ticker, date)] = found

    if not pairs:
        for path in repo_root.rglob("*.csv"):
            if "message" in path.name:
                inferred = _infer_from_filename(path)
                if not inferred:
                    continue
                ticker, date = inferred
                orderbook_candidate = path.with_name(path.name.replace("message", "orderbook"))
                if orderbook_candidate.exists():
                    pairs[(ticker, date)] = (path, orderbook_candidate)

    return [
        (message, orderbook, ticker, date)
        for (ticker, date), (message, orderbook) in sorted(pairs.items())
    ]
