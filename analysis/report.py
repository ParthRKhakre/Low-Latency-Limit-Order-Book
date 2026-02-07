"""Markdown report generation."""
from __future__ import annotations

from pathlib import Path
from typing import Dict


def write_report(
    out_path: Path,
    ticker: str,
    date: str,
    stats: Dict[str, float],
    sanity: Dict[str, int],
    horizons: list[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_prefix = f"{ticker}_{date}"
    plot_files = [
        f"plots/midprice_{plot_prefix}.png",
        f"plots/spread_{plot_prefix}.png",
        f"plots/spread_dist_{plot_prefix}.png",
        f"plots/event_types_{plot_prefix}.png",
        f"plots/events_per_sec_{plot_prefix}.png",
        f"plots/imbalance_future_{plot_prefix}.png",
        f"plots/vol_vs_spread_{plot_prefix}.png",
        f"plots/cancel_submit_{plot_prefix}.png",
    ]
    regime_counts = stats.get("regime_counts", {})
    regime_lines = "\n".join([f"- {k}: {v}" for k, v in regime_counts.items()])

    content = f"""# EDA Report: {ticker} {date}

## Dataset summary
- Rows: {int(stats.get('rows', 0))}
- Time range (sec): {stats.get('time_start', 0):.2f} to {stats.get('time_end', 0):.2f}
- Average spread: {stats.get('avg_spread', 0):.2f}
- Average mid: {stats.get('avg_mid', 0):.2f}

## Sanity checks
- Spread violations: {sanity.get('spread_violations', 0)}
- Non-negative sizes: {sanity.get('sizes_non_negative', 0)}
- Bid monotonic: {sanity.get('bid_monotonic', 0)}
- Ask monotonic: {sanity.get('ask_monotonic', 0)}
- Halt rows: {sanity.get('halt_rows', 0)}

## Key EDA findings
- Spread and midprice dynamics are shown in the plots below.
- Event type distribution highlights liquidity dynamics (submissions vs cancellations).
- Imbalance/future return plot provides a directional signal proxy at H=1 event horizon.
- Volatility vs spread highlights liquidity regimes.

## Regime segmentation counts
{regime_lines or '- Not available'}

## Plots
""" + "\n".join([f"- {plot}" for plot in plot_files]) + f"""

## How this feeds the project
1) Exported features include: best bid/ask, mid, spread, imbalance (L1/topK), depth, microprice,
   event rate, order flow imbalance, and realized volatility.
2) Recommended ML horizons: {', '.join(str(h) for h in horizons)} events, balancing fast reaction
   (short horizon) and stability (longer horizon).
3) Market making is typically most favorable in tight spread and low/normal volatility regimes.
"""

    out_path.write_text(content, encoding="utf-8")
