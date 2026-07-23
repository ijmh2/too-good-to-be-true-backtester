"""Phase 0 end-to-end smoke test.

Fetches SPY (falls back to synthetic data if Yahoo is unreachable/rate-limiting), runs a
buy-and-hold backtest through the engine, and prints the headline metrics. This proves the
whole Phase 0 pipeline — data -> weights -> causal engine -> costs -> metrics — hangs
together before any strategy or validation logic is built on top.
"""

from __future__ import annotations

from tgtbt.data import get_prices_or_fallback
from tgtbt.strategies import BuyAndHold


def main() -> None:
    prices, source = get_prices_or_fallback("SPY", start="2015-01-01", end="2024-12-31", min_rows=100)
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
