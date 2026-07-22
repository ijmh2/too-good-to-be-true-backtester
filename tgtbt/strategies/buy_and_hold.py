"""Buy-and-hold: the benchmark every other strategy has to beat.

Holds a fixed set of assets at fixed weights (equal-weight by default) for the whole period.
Turnover is incurred only on the first day to put the book on, so net and gross returns are
almost identical — exactly what you want from a passive baseline.
"""

from __future__ import annotations

import pandas as pd

from tgtbt.strategies.base import Strategy


class BuyAndHold(Strategy):
    def __init__(self, weights: dict[str, float] | None = None):
        """`weights` maps ticker -> weight. If omitted, equal-weight all columns."""
        self.fixed_weights = weights
        self.name = "buy_and_hold"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        cols = list(prices.columns)
        if self.fixed_weights is None:
            row = {c: 1.0 / len(cols) for c in cols}
        else:
            row = {c: float(self.fixed_weights.get(c, 0.0)) for c in cols}
        return pd.DataFrame([row] * len(prices), index=prices.index, columns=cols)
