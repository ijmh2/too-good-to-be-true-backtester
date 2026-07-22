"""Permutation / random-timing null.

Question: is the strategy's *timing* real, or would any position series with the same
exposure and turnover have done as well by luck? To answer, we keep the strategy's own
weights — so exposure, leverage and turnover are held fixed — but destroy their alignment
with returns, then recompute performance many times to build a null distribution.

- method="circular": roll the whole weight series by a random offset. Preserves the
  autocorrelation and turnover of the positions (up to the single wrap-around seam); only the
  phase relative to returns is randomised. This is the honest "random-timing" null.
- method="shuffle": independently permute the weight rows (a harsher null that also breaks
  position autocorrelation).

The p-value is the share of null runs whose Sharpe matches or beats the real one. A small
p-value means the real timing is doing something a random-timing strategy can't.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tgtbt import metrics
from tgtbt.costs import CostModel
from tgtbt.data import to_returns
from tgtbt.engine import Backtest
from tgtbt.strategies.base import Strategy


@dataclass
class PermutationResult:
    real_sharpe: float
    null_sharpes: np.ndarray
    p_value: float
    method: str

    @property
    def percentile(self) -> float:
        """Where the real Sharpe sits within the null distribution (0-100)."""
        return float((self.null_sharpes < self.real_sharpe).mean() * 100)


def permutation_test(
    strategy: Strategy,
    prices: pd.DataFrame,
    n: int = 1000,
    method: str = "circular",
    cost_model: CostModel | None = None,
    seed: int = 0,
) -> PermutationResult:
    returns = to_returns(prices)
    engine = Backtest(returns, cost_model=cost_model)

    weights = strategy.generate_weights(prices).reindex(
        index=returns.index, columns=returns.columns
    ).fillna(0.0)
    w = weights.to_numpy()
    T = w.shape[0]

    real_sharpe = metrics.sharpe(engine.run(weights).net_returns)

    rng = np.random.default_rng(seed)
    null = np.empty(n)
    for i in range(n):
        if method == "circular":
            shift = int(rng.integers(1, T))
            perm = np.roll(w, shift, axis=0)
        elif method == "shuffle":
            perm = w[rng.permutation(T)]
        else:
            raise ValueError(f"unknown method {method!r}")
        perm_w = pd.DataFrame(perm, index=weights.index, columns=weights.columns)
        null[i] = metrics.sharpe(engine.run(perm_w).net_returns)

    # +1 in numerator/denominator: the observed statistic is itself one possible arrangement.
    p_value = (1 + int(np.sum(null >= real_sharpe))) / (n + 1)
    return PermutationResult(real_sharpe, null, p_value, method)
