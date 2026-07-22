"""Causal, vectorised backtest engine.

The engine owns the clock. A strategy produces **target weights** indexed by date, where
`weights.loc[t]` is the position decided using information available *through the close of
day t*. The engine then earns that position over the *next* day's return:

    portfolio_return[t] = sum_i  weights[t-1, i] * asset_return[t, i]

i.e. weights are shifted forward one bar before being multiplied by returns. This is what
makes look-ahead structurally impossible: a weight decided at t can only ever be paid the
return realised strictly after t. `tests/test_engine.py` locks this property in.

Transaction costs are charged when the book is *established* (at t-1) and therefore reduce
the return of the period they fund (t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tgtbt import TRADING_DAYS
from tgtbt.costs import CostModel


@dataclass
class BacktestResult:
    gross_returns: pd.Series  # daily portfolio returns before costs
    net_returns: pd.Series    # daily portfolio returns after transaction costs
    weights: pd.DataFrame     # target weights actually used (aligned to returns)
    turnover: pd.Series       # one-way turnover per day
    cost_model: CostModel

    @property
    def equity_curve(self) -> pd.Series:
        """Growth of 1 unit, net of costs."""
        return (1.0 + self.net_returns).cumprod()

    def summary(self) -> dict:
        from tgtbt import metrics

        m = metrics.summary(self.net_returns)
        m["avg_turnover"] = float(self.turnover.mean())
        m["total_cost"] = float((self.gross_returns - self.net_returns).sum())
        return m


class Backtest:
    """Run target weights against asset returns with a cost model.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily simple returns, rows=dates, cols=assets.
    cost_model : CostModel, optional
        Defaults to 5 bps per unit one-way turnover.
    """

    def __init__(self, returns: pd.DataFrame, cost_model: CostModel | None = None):
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a DataFrame (one column per asset)")
        self.returns = returns.sort_index()
        self.cost_model = cost_model or CostModel()

    def run(self, weights: pd.DataFrame) -> BacktestResult:
        # Align weights to the return grid. Any date/asset without a stated weight is flat.
        w = weights.reindex(index=self.returns.index, columns=self.returns.columns)
        w = w.fillna(0.0)

        # --- the causal step: yesterday's decision earns today's return ---
        applied = w.shift(1).fillna(0.0)
        gross = (applied * self.returns).sum(axis=1)

        # Costs are incurred when weights change (at decision time t) and are paid out of
        # the period those weights fund (t+1) -> shift the cost forward to match `gross`.
        cost = self.cost_model.cost_series(w).shift(1).fillna(0.0)
        net = gross - cost

        turnover = self.cost_model.turnover(w)

        return BacktestResult(
            gross_returns=gross,
            net_returns=net,
            weights=w,
            turnover=turnover,
            cost_model=self.cost_model,
        )
