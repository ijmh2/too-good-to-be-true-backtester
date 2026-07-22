"""Short-horizon mean reversion ("buy the dip").

Go long a single asset when it is stretched below its own short moving average (z-score below
a threshold), sit in cash otherwise. This family is a classic overfit trap: it often shows a
gorgeous in-sample equity curve, then most of the edge evaporates once realistic transaction
costs meet its high turnover — exactly the kind of thing the scorecard is built to expose.
"""

from __future__ import annotations

import pandas as pd

from tgtbt.strategies.base import Strategy


class MeanReversion(Strategy):
    def __init__(self, asset: str | None = None, lookback: int = 5, entry_z: float = -1.0):
        self.asset = asset
        self.lookback = lookback
        self.entry_z = entry_z
        self.name = f"mean_reversion(lb={lookback},z<{entry_z})"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col = self.asset or prices.columns[0]
        px = prices[col]
        ma = px.rolling(self.lookback, min_periods=self.lookback).mean()
        sd = px.rolling(self.lookback, min_periods=self.lookback).std()
        z = (px - ma) / sd
        weight = (z < self.entry_z).astype(float).fillna(0.0)
        return weight.to_frame(col).reindex(columns=prices.columns).fillna(0.0)


def make_mean_reversion(**params) -> MeanReversion:
    return MeanReversion(**params)
