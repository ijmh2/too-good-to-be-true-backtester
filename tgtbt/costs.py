"""Transaction-cost model.

Costs are charged on **turnover** — the sum of absolute weight changes when the portfolio
rebalances. One unit of turnover = trading 100% of capital one way. `per_side_bps` is the
round-trip-relevant cost applied per unit of one-way turnover, so a full round trip
(buy then later sell, turnover 2.0) costs 2 * per_side_bps.

Defaults are deliberately realistic-to-conservative for liquid US equities/ETFs:
5 bps captures a tight spread plus commission. Bump it for less liquid names.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    per_side_bps: float = 5.0  # cost per unit of one-way turnover, in basis points

    @property
    def rate(self) -> float:
        return self.per_side_bps / 1e4

    def turnover(self, weights: pd.DataFrame) -> pd.Series:
        """One-way turnover per date: sum |w_t - w_{t-1}| across assets.

        The first row is charged against a starting cash position (weights of 0), i.e. the
        cost of putting the initial book on.
        """
        prev = weights.shift(1).fillna(0.0)
        return (weights - prev).abs().sum(axis=1)

    def cost_series(self, weights: pd.DataFrame) -> pd.Series:
        """Per-date cost as a fraction of capital, indexed like `weights`."""
        return self.turnover(weights) * self.rate
