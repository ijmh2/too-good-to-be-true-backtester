"""Tests for the validation layer.

The most important behavioural checks here are on the overfit diagnostics themselves:
- PBO must sit near 0.5 for pure noise (in-sample winners are random out-of-sample) and
  drop toward 0 when one configuration genuinely dominates.
- The deflated Sharpe benchmark must rise with the number of trials.
"""

import numpy as np
import pandas as pd

from tgtbt.costs import CostModel
from tgtbt.data import synthetic_prices
from tgtbt.strategies.trend import make_trend_vt, TrendVolTarget
from tgtbt.strategies.buy_and_hold import BuyAndHold
from tgtbt.strategies.dual_momentum import DualMomentum
from tgtbt.strategies.bollinger_breakout import BollingerBreakout
from tgtbt.validation import (
    expand_grid,
    train_test_split,
    walk_forward,
    parameter_surface,
    permutation_test,
    block_bootstrap,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    cscv_pbo,
    config_returns,
)
from tgtbt.validation.deflated import deflated_sharpe_from_trials

FREE = CostModel(0.0)


def _prices(seed=0, n=1200):
    return synthetic_prices("X", n_days=n, seed=seed)


def test_expand_grid_and_split():
    combos = expand_grid({"a": [1, 2], "b": [3, 4]})
    assert len(combos) == 4 and {"a": 2, "b": 3} in combos
    px = _prices()
    tr, te = train_test_split(px, px.index[600])
    assert tr.index[-1] <= px.index[600] < te.index[0]
    assert len(tr) + len(te) == len(px)


def test_trend_weights_are_valid_and_bounded():
    px = _prices()
    w = TrendVolTarget(trend_window=100, max_leverage=1.0).generate_weights(px)
    assert not w.isna().any().any()
    assert (w >= 0).all().all() and (w <= 1.0 + 1e-9).all().all()
    assert (w.iloc[:99] == 0).all().all()  # 100-day MA valid only from the 100th row on


def test_bollinger_breakout_weights_are_valid_and_bounded():
    px = _prices()
    w = BollingerBreakout(window=20, k=2.0).generate_weights(px)
    assert not w.isna().any().any()
    assert (w >= -1.0 - 1e-9).all().all() and (w <= 1.0 + 1e-9).all().all()
    assert (w.iloc[:19] == 0).all().all()  # 20-day mean/std valid only from the 20th row on


def test_walk_forward_runs_and_is_causal():
    px = _prices()
    grid = {"trend_window": [50, 150], "vol_window": [20], "target_vol": [0.15]}
    wf = walk_forward(make_trend_vt, grid, px, n_folds=3, cost_model=FREE)
    assert len(wf.fold_records) == 3
    assert wf.oos_returns.index.is_monotonic_increasing
    # OOS pieces must not overlap in time.
    assert wf.oos_returns.index.is_unique


def test_parameter_surface_shape():
    px = _prices()
    surf = parameter_surface(
        make_trend_vt, {"trend_window": [50, 100, 150], "vol_window": [20, 40]}, px, cost_model=FREE
    )
    assert len(surf) == 6 and {"trend_window", "vol_window", "score"} <= set(surf.columns)


def test_permutation_pvalue_in_range():
    px = _prices()
    res = permutation_test(BuyAndHold(), px, n=200, cost_model=FREE, seed=1)
    assert 0.0 < res.p_value <= 1.0
    assert res.null_sharpes.shape == (200,)


def test_bootstrap_ci_ordering():
    px = _prices()
    r = BuyAndHold().backtest(px, cost_model=FREE).net_returns
    bs = block_bootstrap(r, n=300, block=20, seed=0)
    lo, hi = bs.ci("sharpe", 0.9)
    assert lo <= bs.point_estimates["sharpe"] * 0 + hi  # lo <= hi
    assert 0.0 <= bs.prob_positive("sharpe") <= 1.0


def test_expected_max_sharpe_monotonic():
    a = expected_max_sharpe(10, 0.01)
    b = expected_max_sharpe(200, 0.01)
    assert b > a > 0  # more trials -> higher luck benchmark


def test_psr_high_for_strong_positive_series():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.005, 2000))  # Sharpe ~ 0.2/day-ish, very strong
    assert probabilistic_sharpe_ratio(r, 0.0) > 0.99


def test_pbo_near_half_for_noise():
    rng = np.random.default_rng(0)
    # 20 configs of pure iid noise: no real edge -> IS winner random OOS -> PBO ~ 0.5.
    M = pd.DataFrame(rng.normal(0, 0.01, size=(1000, 20)))
    res = cscv_pbo(M, n_splits=10)
    assert 0.3 < res.pbo < 0.7


def test_pbo_low_when_one_config_dominates():
    rng = np.random.default_rng(1)
    M = rng.normal(0, 0.01, size=(1000, 20))
    M[:, 0] += 0.004  # config 0 has a large, real drift in every period
    res = cscv_pbo(pd.DataFrame(M), n_splits=10)
    assert res.pbo < 0.2 and res.verdict == "holds up"


def test_dual_momentum_handles_warmup_and_is_single_asset():
    px = pd.concat(
        [synthetic_prices("A", 500, seed=1), synthetic_prices("B", 500, seed=2)], axis=1
    )
    w = DualMomentum(lookback=126).generate_weights(px)
    assert not w.isna().any().any()
    assert (w.iloc[:126] == 0).all().all()          # all-NaN warm-up -> cash, no crash
    assert ((w > 0).sum(axis=1) <= 1).all()          # at most one asset held at a time
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()       # never levered


def test_dual_momentum_ties_do_not_lever():
    # Two identical price columns tie exactly every day -> must split 50/50, never sum to 2.
    base = synthetic_prices("A", 400, seed=1)["A"]
    px = pd.DataFrame({"A": base.values, "B": base.values}, index=base.index)
    w = DualMomentum(lookback=100).generate_weights(px)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()
    invested = w.sum(axis=1) > 0
    assert (w[invested].sum(axis=1).round(6) == 1.0).all()  # fully invested rows sum to 1


def test_deflated_from_trials_keys():
    px = _prices()
    mat = config_returns(make_trend_vt, {"trend_window": [50, 100, 150]}, px, FREE)
    d = deflated_sharpe_from_trials(mat)
    assert {"selected", "n_trials", "sr_star", "psr_vs_zero", "dsr"} <= set(d)
    assert d["n_trials"] == 3
    assert d["dsr"] <= d["psr_vs_zero"] + 1e-9  # deflation can only lower the probability
