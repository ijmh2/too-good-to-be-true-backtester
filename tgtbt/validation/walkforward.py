"""Walk-forward analysis: re-select parameters as time moves on, and only ever *use* a
configuration on data it was never chosen on.

The timeline is cut into contiguous folds. Fold 0 seeds the training set; then for each
subsequent fold we (a) pick the best parameters on everything seen so far (anchored/expanding
window), and (b) apply exactly those parameters to the next, unseen fold and keep those
returns. Stitching the unseen-fold returns together gives an equity curve that no single
parameter choice was fitted to — the honest answer to "would this have worked live?".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tgtbt import metrics
from tgtbt.costs import CostModel
from tgtbt.validation.gridutil import StrategyFactory, expand_grid


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series               # stitched out-of-sample daily returns
    fold_records: list[dict] = field(default_factory=list)  # per-fold selection + scores

    @property
    def oos_sharpe(self) -> float:
        return metrics.sharpe(self.oos_returns)

    def summary(self) -> dict:
        s = metrics.summary(self.oos_returns)
        s["n_folds"] = len(self.fold_records)
        s["params_stable"] = len({tuple(sorted(r["params"].items())) for r in self.fold_records}) == 1
        return s


def walk_forward(
    factory: StrategyFactory,
    grid: dict[str, list],
    prices: pd.DataFrame,
    n_folds: int = 5,
    cost_model: CostModel | None = None,
) -> WalkForwardResult:
    """Anchored walk-forward with per-fold parameter re-selection (best net Sharpe on train)."""
    combos = expand_grid(grid)
    idx = prices.index
    # Split the index into n_folds+1 near-equal contiguous blocks; block 0 = initial train.
    bounds = np.linspace(0, len(idx), n_folds + 2, dtype=int)
    fold_records: list[dict] = []
    oos_pieces: list[pd.Series] = []

    for i in range(1, n_folds + 1):
        train_end = idx[bounds[i] - 1]
        test_start, test_end = idx[bounds[i]], idx[bounds[i + 1] - 1]

        train_prices = prices.loc[:train_end]

        # Select the configuration with the best net Sharpe on the anchored training window.
        best_params, best_score = None, -np.inf
        for params in combos:
            r = factory(**params).backtest(train_prices, cost_model=cost_model).net_returns
            s = metrics.sharpe(r)
            if np.isfinite(s) and s > best_score:
                best_params, best_score = params, s
        best_params = best_params or combos[0]

        # Apply the winner to the *unseen* test fold. History up to test_end feeds the
        # indicators (all backward-looking), but we only keep returns inside the test window.
        applied = factory(**best_params).backtest(
            prices.loc[:test_end], cost_model=cost_model
        ).net_returns
        oos_piece = applied.loc[test_start:test_end]
        oos_pieces.append(oos_piece)

        fold_records.append(
            {
                "fold": i,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "params": best_params,
                "train_sharpe": float(best_score),
                "test_sharpe": float(metrics.sharpe(oos_piece)),
            }
        )

    oos_returns = pd.concat(oos_pieces) if oos_pieces else pd.Series(dtype=float)
    return WalkForwardResult(oos_returns=oos_returns, fold_records=fold_records)
