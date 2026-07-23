"""Batch-test one strategy across a basket of tickers.

Generalizes the manual "does this edge survive on a different asset?" check — run the
identical strategy/grid across several tickers and get a sorted comparison table, rather than
running `run_scorecard` by hand once per ticker.

By default this reruns the Bollinger-breakout finding from the README/session history (real
edge on TSLA, inconclusive on SPY) across a small basket of high-momentum and diversified
names, to see whether the TSLA result generalizes to other volatile single names or was
TSLA-specific.
"""

from __future__ import annotations

from tgtbt.batch import run_batch
from tgtbt.strategies.bollinger_breakout import BollingerBreakout, make_bollinger_breakout

TICKERS = ["TSLA", "NVDA", "AAPL", "SPY", "QQQ"]


def main() -> None:
    grid = {"window": [10, 20, 30, 50], "k": [1.5, 2.0, 2.5]}
    df = run_batch(
        make_bollinger_breakout,
        headline_params={"window": 20, "k": 2.0},
        grid=grid,
        tickers=TICKERS,
        start="2015-01-01",
        end="2024-12-31",
        split_date="2021-12-31",
        perm_n=300,
        boot_n=300,
        n_folds=4,
    )
    print("\n" + df.to_string(float_format=lambda x: f"{x:.3f}"))
    df.to_csv("batch_bollinger_breakout.csv")
    print("\n[saved] batch_bollinger_breakout.csv")


if __name__ == "__main__":
    main()
