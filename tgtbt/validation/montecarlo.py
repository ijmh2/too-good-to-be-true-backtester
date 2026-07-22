"""Monte-Carlo confidence intervals via the stationary/block bootstrap.

A single equity curve is one draw from a noisy process; its Sharpe and max drawdown are
point estimates with real uncertainty. We resample the return series in *blocks* (to keep
short-horizon autocorrelation intact) to generate many alternative histories, then read off
the distribution of each metric. If the strategy's headline Sharpe sits comfortably inside a
cloud that also contains zero, the "edge" is within the noise.

We also return percentile bands of the resampled equity curves for a confidence cone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tgtbt import metrics


@dataclass
class BootstrapResult:
    metric_samples: dict[str, np.ndarray]     # metric name -> array of bootstrap draws
    point_estimates: dict[str, float]         # metric name -> value on the real series
    equity_bands: pd.DataFrame                # percentile bands of resampled equity curves

    def ci(self, metric: str, level: float = 0.95) -> tuple[float, float]:
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        s = self.metric_samples[metric]
        return (float(np.quantile(s, lo)), float(np.quantile(s, hi)))

    def prob_positive(self, metric: str = "sharpe") -> float:
        return float((self.metric_samples[metric] > 0).mean())


def block_bootstrap(
    returns: pd.Series,
    n: int = 1000,
    block: int = 20,
    seed: int = 0,
) -> BootstrapResult:
    r = returns.dropna().to_numpy()
    T = len(r)
    if T < block * 2:
        raise ValueError("series too short for the requested block size")

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))

    funcs = {
        "sharpe": metrics.sharpe,
        "cagr": metrics.cagr,
        "max_drawdown": metrics.max_drawdown,
    }
    samples = {k: np.empty(n) for k in funcs}
    equity_paths = np.empty((n, T))

    for i in range(n):
        # Circular block bootstrap: pick random block start points, concatenate, trim to T.
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        draw = pd.Series(r[idx])
        for k, fn in funcs.items():
            samples[k][i] = fn(draw)
        equity_paths[i] = np.cumprod(1.0 + draw.to_numpy())

    pct = [5, 25, 50, 75, 95]
    bands = pd.DataFrame(
        {f"p{p}": np.percentile(equity_paths, p, axis=0) for p in pct}
    )

    point = {k: float(fn(returns)) for k, fn in funcs.items()}
    return BootstrapResult(samples, point, bands)
