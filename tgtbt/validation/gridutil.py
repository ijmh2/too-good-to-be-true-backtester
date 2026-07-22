"""Shared plumbing for the validation tools.

A *strategy factory* is any callable `factory(**params) -> Strategy`. A *grid* is a dict of
`param_name -> list of values`. Together they define the search space that walk-forward,
the robustness surface, and CPCV all explore — the same "how many things did we try?" that
the deflated Sharpe later charges us for.
"""

from __future__ import annotations

import itertools
from typing import Callable

import pandas as pd

from tgtbt import metrics
from tgtbt.costs import CostModel
from tgtbt.strategies.base import Strategy

StrategyFactory = Callable[..., Strategy]


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a parameter grid -> list of param dicts."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def evaluate(
    factory: StrategyFactory,
    params: dict,
    prices: pd.DataFrame,
    cost_model: CostModel | None = None,
) -> pd.Series:
    """Net daily returns of one configuration on one price set."""
    return factory(**params).backtest(prices, cost_model=cost_model).net_returns


def config_returns(
    factory: StrategyFactory,
    grid: dict[str, list],
    prices: pd.DataFrame,
    cost_model: CostModel | None = None,
) -> pd.DataFrame:
    """Net-return matrix: one column per configuration in the grid.

    Column labels are the repr of each param dict. This matrix is the raw material for the
    robustness surface and for CPCV.
    """
    out = {}
    for params in expand_grid(grid):
        label = ", ".join(f"{k}={v}" for k, v in params.items())
        out[label] = evaluate(factory, params, prices, cost_model)
    return pd.DataFrame(out)


def score(returns: pd.Series) -> float:
    """Default selection score = net Sharpe. Central so every tool ranks the same way."""
    return metrics.sharpe(returns)
