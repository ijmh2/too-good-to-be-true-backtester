"""Dual momentum: cross-asset rotation with an absolute-momentum safety switch.

Across a small menu of assets, hold the one with the strongest trailing return (relative
momentum) — but only if that return is positive (absolute momentum); otherwise sit in cash.
The idea (Antonacci) is to ride the leading asset while stepping aside in broad downturns.
Needs a multi-column price frame.
"""

from __future__ import annotations

import pandas as pd

from tgtbt.strategies.base import Strategy


class DualMomentum(Strategy):
    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.name = f"dual_momentum(lb={lookback})"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        mom = prices / prices.shift(self.lookback) - 1.0        # trailing return per asset
        row_max = mom.max(axis=1)                               # best trailing return per day
        # One-hot the relative-momentum winner, but only if it clears absolute momentum (>0).
        # All-NaN warm-up rows give row_max=NaN -> no match -> stay in cash (all-zero row).
        winner = mom.eq(row_max, axis=0) & mom.gt(0)
        winner = winner.astype(float)
        # Exact ties (two assets sharing the max) would otherwise sum to >1 and lever up;
        # split the allocation equally so each row's weights sum to at most 1.
        row_sums = winner.sum(axis=1)
        weights = winner.div(row_sums.where(row_sums != 0, 1.0), axis=0)
        return weights.reindex(columns=prices.columns).fillna(0.0)


def make_dual_momentum(**params) -> DualMomentum:
    return DualMomentum(**params)
