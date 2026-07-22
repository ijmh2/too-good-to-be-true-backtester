"""Phase 4 end-to-end: run the overfit scorecard on real strategies and save the figures.

Two contrasting cases, both judged by the identical gauntlet:
  1. Trend + vol-target on SPY  — a strategy with genuine academic backing.
  2. Short-horizon mean reversion on SPY — a classic in-sample-pretty, cost-fragile trap.

The point of the repo is that the *framework*, not the author, decides which is which.
Figures are written to docs/ (committed) and a markdown verdict is printed for each.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tgtbt.data import get_prices, synthetic_prices
from tgtbt.strategies import (
    BuyAndHold,
    TrendVolTarget,
    make_trend_vt,
    MeanReversion,
    make_mean_reversion,
)
from tgtbt.reporting.scorecard import run_scorecard
from tgtbt.reporting import style as S

DOCS = Path(__file__).resolve().parent.parent / "docs"
SPLIT = "2021-12-31"  # in-sample <= 2021, out-of-sample 2022+


def load_spy() -> pd.DataFrame:
    try:
        px = get_prices("SPY", start="2010-01-01", end="2024-12-31")
        if px.dropna(how="all").shape[0] > 500:
            return px
        raise RuntimeError("insufficient rows")
    except Exception as exc:  # noqa: BLE001
        print(f"[data] live fetch failed ({exc!r}); using synthetic prices.")
        return synthetic_prices("SPY", n_days=3000, seed=7)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    S.apply_style()
    prices = load_spy()
    benchmark = BuyAndHold().backtest(prices).net_returns
    print(f"[data] SPY rows={len(prices)}, {prices.index[0].date()}..{prices.index[-1].date()}\n")

    cases = [
        (
            "trend",
            TrendVolTarget(trend_window=200, vol_window=20, target_vol=0.15),
            make_trend_vt,
            {"trend_window": [50, 100, 150, 200], "vol_window": [20, 40, 60], "target_vol": [0.15]},
        ),
        (
            "meanrev",
            MeanReversion(lookback=5, entry_z=-1.0),
            make_mean_reversion,
            {"lookback": [3, 5, 10, 20], "entry_z": [-0.5, -1.0, -1.5, -2.0]},
        ),
    ]

    for tag, strat, factory, grid in cases:
        print(f"=== {strat.name} ===")
        card = run_scorecard(
            strat, factory, grid, prices, benchmark=benchmark, split_date=SPLIT,
            perm_n=500, boot_n=500, n_folds=5,
        )
        fig = card.figure()
        out = DOCS / f"scorecard_{tag}.png"
        S.save_fig(fig, str(out))
        print(card.to_markdown())
        print(f"\n[saved] {out}\n")


if __name__ == "__main__":
    main()
