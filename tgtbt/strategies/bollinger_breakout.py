"""Bollinger Band breakout / trend-continuation.

Go long when price closes above the upper band (rolling mean + k * rolling std) — a
volatility breakout read as the start of a trend — and hold until price closes back below
the rolling mean, at which point we flip flat. This is a classic momentum-ignition setup:
it profits when breakouts are followed by continuation, and gives back edge whenever bands
are just noise (the harness's job is to tell the two apart).

The mean/std at t use only price data up to and including t (`rolling(window)`, never
centred), so the point-in-time contract holds. The entry/exit rule is inherently
path-dependent (position persists between the breakout and the mean-reversion exit), which
is handled with an event series that is forward-filled rather than a stateful loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tgtbt.strategies.base import Strategy


class BollingerBreakout(Strategy):
    def __init__(self, asset: str | None = None, window: int = 20, k: float = 2.0):
        self.asset = asset
        self.window = window
        self.k = k
        self.name = f"bollinger_breakout(window={window},k={k})"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col = self.asset or prices.columns[0]
        px = prices[col]

        ma = px.rolling(self.window, min_periods=self.window).mean()
        sd = px.rolling(self.window, min_periods=self.window).std()
        upper = ma + self.k * sd

        entry = px > upper  # volatility breakout -> go long
        exit_ = px < ma  # back below the rolling mean -> flat / cash

        # Event series: 1.0 on entry days, 0.0 on exit days, NaN otherwise. Forward-filling
        # this turns discrete entry/exit events into a persistent position, and the trailing
        # fillna(0.0) keeps both the pre-entry and warm-up periods flat.
        signal = pd.Series(np.nan, index=px.index)
        signal[entry] = 1.0
        signal[exit_] = 0.0
        weight = signal.ffill().fillna(0.0)

        return weight.to_frame(col).reindex(columns=prices.columns).fillna(0.0)


def make_bollinger_breakout(**params) -> BollingerBreakout:
    """Factory for use with the validation tools (walk-forward, robustness, CPCV)."""
    return BollingerBreakout(**params)
