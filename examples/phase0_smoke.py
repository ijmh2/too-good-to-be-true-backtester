"""Phase 0 end-to-end smoke test.

Fetches SPY (falls back to synthetic data if Yahoo is unreachable/rate-limiting), runs a
buy-and-hold backtest through the engine, and prints the headline metrics. This proves the
whole Phase 0 pipeline — data -> weights -> causal engine -> costs -> metrics — hangs
together before any strategy or validation logic is built on top.
"""

from __future__ import annotations

import pandas as pd

from tgtbt.data import get_prices, synthetic_prices
from tgtbt.strategies import BuyAndHold


def load_prices() -> tuple[pd.DataFrame, str]:
    try:
        prices = get_prices("SPY", start="2015-01-01", end="2024-12-31")
        if prices.dropna(how="all").shape[0] < 100:
            raise RuntimeError("too few rows")
        return prices, "live SPY (yfinance)"
    except Exception as exc:  # noqa: BLE001 - offline fallback is the whole point
        print(f"[data] live fetch failed ({exc!r}); using synthetic prices instead.")
        return synthetic_prices("SPY", n_days=2000, seed=42), "synthetic GBM"


def main() -> None:
    prices, source = load_prices()
    print(f"[data] source = {source}, rows = {len(prices)}, "
          f"span = {prices.index[0].date()} .. {prices.index[-1].date()}")

    result = BuyAndHold().backtest(prices)
    stats = result.summary()

    print("\nBuy & Hold — net of 5bps costs")
    print("-" * 34)
    for k, v in stats.items():
        print(f"{k:>14} : {v:,.4f}" if isinstance(v, float) else f"{k:>14} : {v}")

    final = result.equity_curve.iloc[-1]
    print(f"\n1.0 invested -> {final:,.2f}")


if __name__ == "__main__":
    main()
