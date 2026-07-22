"""Trend-following with volatility targeting — the first candidate "edge".

Two ideas stacked, both well-supported in the literature:

1. **Trend filter.** Hold the asset only when its price is above its own moving average
   (a time-series-momentum proxy). Sit in cash otherwise. This is what historically cuts
   the deep equity drawdowns.
2. **Volatility targeting.** Scale the position so the strategy's *ex-ante* volatility sits
   near a constant target — lever up in calm markets, de-risk in turbulent ones. This is
   what historically improves the Sharpe.

Every input to the weight at date t (moving average, realised vol) uses a *backward* rolling
window ending at t, so the point-in-time contract holds and the engine's shift(1) does the
rest. Single-asset by design; `asset` picks the column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tgtbt import TRADING_DAYS
from tgtbt.strategies.base import Strategy


class TrendVolTarget(Strategy):
    def __init__(
        self,
        asset: str | None = None,
        trend_window: int = 200,
        vol_window: int = 20,
        target_vol: float = 0.15,
        max_leverage: float = 1.0,
    ):
        self.asset = asset
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.name = f"trend_vt(ma={trend_window},vol={vol_window},tv={target_vol})"

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        col = self.asset or prices.columns[0]
        px = prices[col]

        # 1. Trend filter: 1.0 when above the moving average, else 0.0 (flat / in cash).
        ma = px.rolling(self.trend_window, min_periods=self.trend_window).mean()
        long_signal = (px > ma).astype(float)

        # 2. Volatility target: scale by target_vol / realised_vol, capped at max_leverage.
        rets = px.pct_change()
        realised_vol = rets.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(
            TRADING_DAYS
        )
        scale = (self.target_vol / realised_vol).clip(upper=self.max_leverage)

        weight = (long_signal * scale).fillna(0.0)
        return weight.to_frame(col).reindex(columns=prices.columns).fillna(0.0)


def make_trend_vt(**params) -> TrendVolTarget:
    """Factory for use with the validation tools (walk-forward, robustness, CPCV)."""
    return TrendVolTarget(**params)
