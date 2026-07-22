"""Parameter-robustness surface.

The single most intuitive overfit test: sweep the parameters and look at the shape of the
performance surface. A *real* edge sits on a broad plateau — nearby parameter values work
almost as well, because the effect is genuine. An *overfit* edge is a lonely spike:
performance craters the moment you nudge a parameter, because you tuned to noise.

`parameter_surface` returns tidy long-form results; `to_pivot` reshapes a two-parameter
sweep into a grid ready for a heatmap.
"""

from __future__ import annotations

import pandas as pd

from tgtbt import metrics
from tgtbt.costs import CostModel
from tgtbt.validation.gridutil import StrategyFactory, expand_grid

_METRICS = {
    "sharpe": metrics.sharpe,
    "cagr": metrics.cagr,
    "max_drawdown": metrics.max_drawdown,
    "calmar": metrics.calmar,
}


def parameter_surface(
    factory: StrategyFactory,
    grid: dict[str, list],
    prices: pd.DataFrame,
    metric: str = "sharpe",
    cost_model: CostModel | None = None,
) -> pd.DataFrame:
    """Evaluate `metric` for every parameter combination in `grid`.

    Returns a long-form DataFrame: one row per combo, columns = the swept params + `score`.
    """
    fn = _METRICS[metric]
    rows = []
    for params in expand_grid(grid):
        r = factory(**params).backtest(prices, cost_model=cost_model).net_returns
        rows.append({**params, "score": float(fn(r))})
    out = pd.DataFrame(rows)
    out.attrs["metric"] = metric
    return out


def to_pivot(surface: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    """Reshape a two-parameter surface into an index x columns grid of scores."""
    return surface.pivot(index=index, columns=columns, values="score")
