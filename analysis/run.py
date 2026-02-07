"""CLI entrypoint for LOBSTER analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .clean_validate import add_halt_flags, sanity_checks, write_sanity
from .config import Config
from .discovery import discover_pairs
from .eda import run_eda
from .features import compute_features
from .labels import generate_labels
from .load_lobster import load_lobster
from .report import write_report


def _export(df: pd.DataFrame, path: Path, export_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if export_format == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _process_pair(message_path: Path, orderbook_path: Path, ticker: str, date: str, cfg: Config) -> None:
    chunk_rows = cfg.chunk_rows if message_path.stat().st_size > 200 * 1024 * 1024 else None
    full_df, iterator = load_lobster(message_path, orderbook_path, cfg.num_levels, chunk_rows)
    if iterator is not None:
        frames = [chunk for chunk in iterator]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = full_df
    df = add_halt_flags(df)

    sanity = sanity_checks(df, cfg.num_levels)
    sanity_path = Path(cfg.output_dir) / "sanity" / f"sanity_{ticker}_{date}.json"
    write_sanity(sanity, sanity_path)

    features = compute_features(df, cfg.num_levels, cfg.topk)
    labeled = generate_labels(features, cfg.horizons_events)

    processed_path = Path(cfg.output_dir) / "processed" / f"processed_{ticker}_{date}.{cfg.export_format}"
    _export(labeled, processed_path, cfg.export_format)

    for horizon in cfg.horizons_events:
        cols = [
            "time_ns",
            "event_type",
            "order_id",
            "size",
            "price",
            "direction",
            "best_bid",
            "best_ask",
            "mid",
            "spread",
            "is_halt",
            "level1_imbalance",
            "topk_imbalance",
            "depth_topk_bid",
            "depth_topk_ask",
            "depth_total_topk",
            "microprice",
            "event_rate",
            "order_flow_imbalance",
            "realized_volatility",
            f"raw_move_H{horizon}",
            f"label_H{horizon}",
        ]
        feature_df = labeled[cols]
        feature_path = (
            Path(cfg.output_dir)
            / "features"
            / f"features_H{horizon}_{ticker}_{date}.{cfg.export_format}"
        )
        _export(feature_df, feature_path, cfg.export_format)

    if cfg.plotting:
        plot_dir = Path(cfg.output_dir) / "plots"
        stats = run_eda(labeled, plot_dir, ticker, date)
    else:
        stats = {}

    report_path = Path(cfg.output_dir) / "reports" / f"eda_report_{ticker}_{date}.md"
    write_report(report_path, ticker, date, stats, sanity, cfg.horizons_events)


def main() -> None:
    parser = argparse.ArgumentParser(description="LOBSTER analysis runner")
    parser.add_argument("--data_root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--format", type=str, default=None, choices=["parquet", "csv"])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = Config.load(Path("config.yaml"))
    if args.out:
        cfg.output_dir = args.out
    if args.format:
        cfg.export_format = args.format
    if args.no_plots:
        cfg.plotting = False

    pairs = discover_pairs(args.data_root, Path("."))
    if not pairs:
        raise SystemExit("No dataset pairs found.")

    if args.all:
        selected = pairs
    else:
        if not args.ticker or not args.date:
            raise SystemExit("Provide --ticker and --date or use --all.")
        selected = [
            pair
            for pair in pairs
            if pair[2].lower() == args.ticker.lower() and pair[3] == args.date
        ]
        if not selected:
            raise SystemExit("Requested ticker/date not found in discovered pairs.")

    for message_path, orderbook_path, ticker, date in selected:
        _process_pair(message_path, orderbook_path, ticker, date, cfg)


if __name__ == "__main__":
    main()
