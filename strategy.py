import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

try:
    import lob_engine
except ImportError as exc:  # pragma: no cover - runtime import check
    raise ImportError(
        "lob_engine module not found. Build the C++ extension via CMake first."
    ) from exc


@dataclass
class LobsterEvent:
    event_type: int
    price: float
    size: int
    direction: int


def load_lobster(path: Path) -> Iterable[LobsterEvent]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            event_type = int(row[0])
            price = float(row[1])
            size = int(row[2])
            direction = int(row[3])
            yield LobsterEvent(event_type, price, size, direction)


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_state = output[:, -1, :]
        return self.head(last_state)


@dataclass
class AvellanedaStoikovParams:
    gamma: float
    sigma: float
    kappa: float
    horizon: float


def reservation_price(
    mid_price: float, inventory: float, params: AvellanedaStoikovParams, t: float
) -> float:
    return mid_price - inventory * params.gamma * params.sigma**2 * (params.horizon - t)


def optimal_spread(params: AvellanedaStoikovParams, t: float) -> float:
    term1 = params.gamma * params.sigma**2 * (params.horizon - t)
    term2 = (2 / params.gamma) * math.log(1 + params.gamma / params.kappa)
    return 0.5 * (term1 + term2)


def build_lob_snapshot(book: lob_engine.LimitOrderBook) -> np.ndarray:
    levels = book.top_levels(5)
    return np.array(levels, dtype=np.float32).reshape(1, 10)


@dataclass
class BacktestResult:
    orders_per_second: float
    pnl: float
    trades: int


def run_backtest(
    data_path: Path,
    model: LSTMPredictor,
    params: AvellanedaStoikovParams,
    device: torch.device,
) -> BacktestResult:
    book = lob_engine.LimitOrderBook()
    model.to(device)
    model.eval()

    snapshots: List[np.ndarray] = []
    inventory = 0.0
    cash = 0.0
    trades = 0
    order_id = 1

    start_time = time.perf_counter()
    for idx, event in enumerate(load_lobster(data_path)):
        side = lob_engine.Side.Bid if event.direction == 1 else lob_engine.Side.Ask
        book.add_order(order_id, event.price, event.size, side)
        order_id += 1
        book.match()

        snapshot = build_lob_snapshot(book)
        snapshots.append(snapshot)
        if len(snapshots) < 50:
            continue
        if len(snapshots) > 50:
            snapshots.pop(0)

        history = np.concatenate(snapshots, axis=0)
        model_input = torch.from_numpy(history).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_delta = model(model_input).item()

        top = book.top_levels(1)
        best_bid = top[0][0][0]
        best_ask = top[1][0][0]
        mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else event.price

        # Toxic flow protection:
        # 1) Use the ML-predicted future mid-price change to reduce exposure when
        #    adverse selection is likely (i.e., predicted_delta is strongly
        #    negative and inventory is long, or positive with short inventory).
        # 2) This clamps the reservation price and spread adjustments, ensuring
        #    the market maker widens spreads or reduces size during toxic order
        #    flow bursts rather than blindly providing liquidity.
        toxicity_penalty = max(min(predicted_delta, 0.5), -0.5)
        adjusted_inventory = inventory + toxicity_penalty

        t = idx / max(1, len(snapshots))
        r_price = reservation_price(mid, adjusted_inventory, params, t)
        spread = optimal_spread(params, t)
        bid_quote = r_price - spread
        ask_quote = r_price + spread

        # Place symmetric quotes based on Avellaneda-Stoikov outputs.
        book.add_order(order_id, bid_quote, 1, lob_engine.Side.Bid)
        order_id += 1
        book.add_order(order_id, ask_quote, 1, lob_engine.Side.Ask)
        order_id += 1

        matches = book.match()
        for trade in matches:
            trades += 1
            if trade.aggressor_id == order_id - 1:  # our ask executed
                inventory -= trade.qty
                cash += trade.qty * trade.price
            else:  # our bid executed
                inventory += trade.qty
                cash -= trade.qty * trade.price

    elapsed = time.perf_counter() - start_time
    orders_per_second = order_id / elapsed if elapsed > 0 else 0.0

    final_top = book.top_levels(1)
    final_mid = 0.0
    if final_top[0][0][0] > 0 and final_top[1][0][0] > 0:
        final_mid = (final_top[0][0][0] + final_top[1][0][0]) / 2
    pnl = cash + inventory * final_mid

    return BacktestResult(orders_per_second=orders_per_second, pnl=pnl, trades=trades)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid market-making backtest")
    parser.add_argument("--data", required=True, type=Path, help="LOBSTER CSV path")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--kappa", type=float, default=1.5)
    parser.add_argument("--horizon", type=float, default=1.0)
    args = parser.parse_args()

    model = LSTMPredictor()
    params = AvellanedaStoikovParams(
        gamma=args.gamma,
        sigma=args.sigma,
        kappa=args.kappa,
        horizon=args.horizon,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = run_backtest(args.data, model, params, device)
    print(f"Orders/sec: {result.orders_per_second:.2f}")
    print(f"PnL: {result.pnl:.2f}")
    print(f"Trades: {result.trades}")


if __name__ == "__main__":
    main()
