"""Metric sanity checks against hand-computable cases."""

import numpy as np
import pandas as pd

from tgtbt import metrics


def test_max_drawdown_known():
    # +100% then -50% -> back to start; peak-to-trough drawdown is exactly -50%.
    r = pd.Series([1.0, -0.5])
    assert np.isclose(metrics.max_drawdown(r), -0.5)


def test_sharpe_sign_and_zero_vol():
    up = pd.Series([0.001] * 252)
    assert metrics.sharpe(up) > 0
    assert np.isnan(metrics.sharpe(pd.Series([0.0] * 10)))  # zero vol -> undefined


def test_cagr_doubles_in_one_year():
    # 252 daily returns that compound to exactly 2x -> CAGR ~ 100%.
    daily = 2.0 ** (1 / 252) - 1
    r = pd.Series([daily] * 252)
    assert np.isclose(metrics.cagr(r), 1.0, atol=1e-6)


def test_alpha_beta_recovers_line():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0.0004, 0.01, 500))
    r = 0.5 * b + 0.0002  # beta 0.5, small positive daily alpha
    alpha, beta = metrics.alpha_beta(r, b)
    assert np.isclose(beta, 0.5, atol=0.02)
    assert alpha > 0
