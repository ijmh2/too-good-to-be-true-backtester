"""Batch testing: run one strategy across many single-asset tickers, one row per ticker.

Directly automates the check that matters most for catching overfitting-by-asset-selection:
does an "edge" found on one name survive on others, or was it an artefact of that one name's
idiosyncratic history? (This is exactly how the Bollinger-breakout example's "likely real
edge" on TSLA turned out to be "inconclusive" on SPY — this module runs that comparison for
you, across as many tickers as you like, instead of doing it by hand one at a time.)

Only meaningful for single-asset strategies (ones that read `prices.columns[0]` / an `asset`
param), since each ticker is fetched and backtested independently, not as a joint multi-asset
universe. For cross-sectional strategies (like `DualMomentum`) run them normally through
`run_scorecard` on their intended multi-column universe instead.
"""

from __future__ import annotations

import pandas as pd

from tgtbt.costs import CostModel
from tgtbt.data import get_prices_or_fallback
from tgtbt.reporting.scorecard import run_scorecard
from tgtbt.strategies import BuyAndHold
from tgtbt.validation.gridutil import StrategyFactory

_VERDICT_ORDER = {"likely real edge": 0, "inconclusive": 1, "likely overfit": 2}


def run_batch(
    factory: StrategyFactory,
    headline_params: dict,
    grid: dict[str, list],
    tickers: list[str],
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    split_date: str | None = None,
    cost_model: CostModel | None = None,
    perm_n: int = 300,
    boot_n: int = 300,
    n_folds: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the identical strategy/grid on each ticker in `tickers`; one row per ticker.

    `headline_params` builds the specific strategy instance reported on (e.g. the config
    you'd headline in a write-up); `grid` is the search space the validation tools
    (walk-forward, CPCV, deflated Sharpe) sweep for that same ticker's data. Both usually
    match what you'd pass to `run_scorecard` directly for a single ticker.

    Rows are sorted "likely real edge" -> "inconclusive" -> "likely overfit" so the most
    interesting results sort to the top regardless of ticker order.
    """
    cost_model = cost_model or CostModel()
    rows = []
    for ticker in tickers:
        if verbose:
            print(f"[batch] {ticker} ...", end=" ", flush=True)
        prices, source = get_prices_or_fallback(ticker, start=start, end=end)
        strategy = factory(**headline_params)
        benchmark = BuyAndHold().backtest(prices, cost_model=cost_model).net_returns
        card = run_scorecard(
            strategy, factory, grid, prices, benchmark=benchmark, split_date=split_date,
            cost_model=cost_model, perm_n=perm_n, boot_n=boot_n, n_folds=n_folds,
        )
        rows.append(
            {
                "ticker": ticker,
                "verdict": card.verdict,
                "data_source": source,
                "split_date": str(pd.Timestamp(card.split_date).date()),
                "sharpe": card.full_summary["sharpe"],
                "wf_oos_sharpe": card.wf_result.oos_sharpe,
                "perm_p": card.perm_result.p_value,
                "dsr": card.deflated["dsr"],
                "pbo": card.pbo.pbo,
                "max_drawdown": card.full_summary["max_drawdown"],
            }
        )
        if verbose:
            print(card.verdict)

    df = pd.DataFrame(rows).set_index("ticker")
    return df.sort_values(by="verdict", key=lambda s: s.map(_VERDICT_ORDER))
