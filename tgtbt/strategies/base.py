"""Strategy base class.

A strategy maps a price history to **target weights**. The one rule that keeps a backtest
honest:

    weights.loc[t] may depend only on prices up to and including index t.

Row t is the position you would hold *going into* t+1; the engine shifts weights forward one
bar before applying returns, so the decision at t is always paid the return realised after t.
Vectorised strategies (rolling means, etc.) satisfy the rule naturally as long as every
window looks *backward* — never use a centred or forward-looking window.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from tgtbt.costs import CostModel
from tgtbt.data import to_returns
from tgtbt.engine import Backtest, BacktestResult


class Strategy(ABC):
    #: human-readable name used in reports
    name: str = "strategy"

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return target weights (rows=dates, cols=assets), each row using only past/current prices."""

    def backtest(
        self, prices: pd.DataFrame, cost_model: CostModel | None = None
    ) -> BacktestResult:
        """Convenience: turn prices into returns, generate weights, run the engine."""
        weights = self.generate_weights(prices)
        returns = to_returns(prices)
        return Backtest(returns, cost_model=cost_model).run(weights)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"


def equal_weight(mask: pd.DataFrame) -> pd.DataFrame:
    """Normalise a boolean/0-1 selection frame so each row's active weights sum to 1.

    Rows with nothing selected stay all-zero (in cash). Handy for building long-only
    cross-sectional strategies out of a selection signal.
    """
    mask = mask.astype(float)
    row_sums = mask.sum(axis=1)
    return mask.div(row_sums.where(row_sums != 0, 1.0), axis=0)
