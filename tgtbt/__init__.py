"""too-good-to-be-true-backtester (tgtbt).

A strategy-validation harness that treats overfitting as the default hypothesis.
"""

__version__ = "0.0.1"

# Bars-per-year for annualising daily statistics. Trading days, not calendar days.
# Defined before submodule imports below, which read it at import time.
TRADING_DAYS = 252

from tgtbt.costs import CostModel  # noqa: E402
from tgtbt.engine import Backtest, BacktestResult  # noqa: E402
from tgtbt import metrics  # noqa: E402

__all__ = ["Backtest", "BacktestResult", "CostModel", "metrics", "TRADING_DAYS"]
