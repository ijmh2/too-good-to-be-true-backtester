"""Strategy implementations. Import the base class from here."""

from tgtbt.strategies.base import Strategy
from tgtbt.strategies.buy_and_hold import BuyAndHold
from tgtbt.strategies.trend import TrendVolTarget, make_trend_vt
from tgtbt.strategies.mean_reversion import MeanReversion, make_mean_reversion
from tgtbt.strategies.dual_momentum import DualMomentum, make_dual_momentum
from tgtbt.strategies.bollinger_breakout import BollingerBreakout, make_bollinger_breakout

__all__ = [
    "Strategy",
    "BuyAndHold",
    "TrendVolTarget",
    "make_trend_vt",
    "MeanReversion",
    "make_mean_reversion",
    "DualMomentum",
    "make_dual_momentum",
    "BollingerBreakout",
    "make_bollinger_breakout",
]
