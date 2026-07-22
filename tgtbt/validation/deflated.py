"""Probabilistic and Deflated Sharpe Ratios (Bailey & Lopez de Prado).

A Sharpe ratio is an *estimate* — noisy, and biased upward by non-normal returns and by the
number of configurations you tried before reporting your best. Two corrections:

- **Probabilistic Sharpe Ratio (PSR)**: the probability that the true Sharpe exceeds a
  benchmark, given the sample length and the returns' skew/kurtosis. Fat left tails and
  negative skew *lower* it.
- **Deflated Sharpe Ratio (DSR)**: PSR where the benchmark is set to the Sharpe you'd expect
  to achieve *by luck* after N trials. Search over many parameters and the bar rises; a
  Sharpe that looked great across 200 configs may not clear its own deflated benchmark.

All Sharpes here are **per-period** (not annualised); that's the convention the formulas use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

_EULER = 0.5772156649015329


def _per_period_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd else np.nan


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark), adjusting for sample size, skew, kurtosis."""
    r = returns.dropna()
    n = len(r)
    if n < 3:
        return np.nan
    sr = _per_period_sharpe(r)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # pandas .kurt() is excess; formula wants non-excess
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum per-period Sharpe across `n_trials` independent lucky draws.

    var_sharpe is the variance of the per-period Sharpes across the trials tested. This is the
    benchmark the winning strategy must beat to not be explicable as the best of N noise draws.
    """
    if n_trials < 2 or var_sharpe <= 0:
        return 0.0
    sigma = np.sqrt(var_sharpe)
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(sigma * ((1 - _EULER) * z1 + _EULER * z2))


def deflated_sharpe_ratio(
    returns: pd.Series, n_trials: int, var_sharpe: float
) -> float:
    """DSR = PSR benchmarked against the expected max Sharpe from `n_trials` trials."""
    sr_star = expected_max_sharpe(n_trials, var_sharpe)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr_star)


def deflated_sharpe_from_trials(
    trial_returns: pd.DataFrame, selected: str | None = None
) -> dict:
    """Convenience: take the full trial matrix, deflate the (per default, best) config.

    `trial_returns` has one column of net returns per configuration. Returns a dict with the
    PSR, DSR, the deflation benchmark, and the metadata needed to report them honestly.
    """
    per_period = {c: _per_period_sharpe(trial_returns[c]) for c in trial_returns.columns}
    sr_series = pd.Series(per_period).dropna()
    n_trials = int(sr_series.shape[0])
    var_sharpe = float(sr_series.var(ddof=1)) if n_trials > 1 else 0.0

    if selected is None:
        selected = sr_series.idxmax()  # the config a naive researcher would report

    sel_returns = trial_returns[selected]
    sr_star = expected_max_sharpe(n_trials, var_sharpe)
    return {
        "selected": selected,
        "n_trials": n_trials,
        "var_sharpe": var_sharpe,
        "sr_star": sr_star,
        "psr_vs_zero": probabilistic_sharpe_ratio(sel_returns, 0.0),
        "dsr": probabilistic_sharpe_ratio(sel_returns, sr_star),
    }
