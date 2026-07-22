"""Template for a strategy you can upload into the UI.

The UI looks for three module-level names:

    STRATEGY : a tgtbt Strategy instance (the specific configuration to headline)
    FACTORY  : a callable(**params) -> Strategy   (lets the validators re-parameterise it)
    GRID     : dict[str, list]                     (the parameter search space to stress-test)

Optionally:
    TICKERS  : str  (default universe, comma-separated; overridable in the UI)

A strategy only has to implement `generate_weights(prices) -> weights`, using strictly
backward-looking windows so it never peeks at the future. Copy this file, edit the logic,
and upload it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tgtbt.strategies.base import Strategy


class MyStrategy(Strategy):
    """Example: hold the asset only when it's above its N-day moving average."""

    def __init__(self, asset: str | None = None, window: int = 100):
        self.asset = asset
        self.window = window
        self.name = f"my_strategy(window={window})"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col = self.asset or prices.columns[0]
        px = prices[col]
        ma = px.rolling(self.window, min_periods=self.window).mean()
        weight = (px > ma).astype(float).fillna(0.0)   # 1 when above MA, else in cash
        return weight.to_frame(col).reindex(columns=prices.columns).fillna(0.0)


def make(**params) -> MyStrategy:
    return MyStrategy(**params)


# --- the three names the UI imports ---
STRATEGY = MyStrategy(window=100)
FACTORY = make
GRID = {"window": [20, 50, 100, 150, 200]}
TICKERS = "SPY"
