"""Engine correctness — above all, that it cannot look ahead.

The marquee test (`test_no_lookahead_oracle`) hands the engine a clairvoyant strategy that
sets each day's weight from that *same day's* return. A leaky engine would let it capture the
sum of absolute returns (a perfect, impossible equity curve). Our causal engine must instead
pay each weight only the *next* day's return, so the oracle earns far less. That gap is the
proof there's no look-ahead.
"""

import numpy as np
import pandas as pd

from tgtbt.costs import CostModel
from tgtbt.engine import Backtest


def _returns(seed=1, n=300):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame({"A": rng.normal(0.0005, 0.01, n)}, index=idx)


def test_full_weight_reproduces_asset_return():
    """Constant weight 1.0 on one asset -> portfolio gross return equals that asset's return."""
    rets = _returns()
    w = pd.DataFrame(1.0, index=rets.index, columns=rets.columns)
    res = Backtest(rets, CostModel(0.0)).run(w)
    # Day 0 has no prior position (gross 0); from day 1 on it tracks the asset exactly.
    pd.testing.assert_series_equal(
        res.gross_returns.iloc[1:], rets["A"].iloc[1:], check_names=False
    )


def test_no_lookahead_oracle():
    """A same-day oracle must NOT achieve the clairvoyant sum(|returns|)."""
    rets = _returns()
    a = rets["A"]

    # Clairvoyant weights: +1 if today up, -1 if down (uses information from time t).
    oracle = np.sign(a).to_frame("A")
    res = Backtest(rets, CostModel(0.0)).run(oracle)

    clairvoyant = a.abs().sum()               # impossible perfect capture
    causal_expected = (np.sign(a).shift(1) * a).iloc[1:].sum()  # what a t+1 engine earns

    assert np.isclose(res.gross_returns.sum(), causal_expected)
    # And it is strictly, materially worse than clairvoyance -> no leak.
    assert res.gross_returns.sum() < 0.5 * clairvoyant


def test_shift_invariance_of_engine():
    """Feeding weights already shifted by one bar equals letting the engine shift them."""
    rets = _returns(seed=7)
    rng = np.random.default_rng(0)
    w = pd.DataFrame(rng.choice([0.0, 1.0], size=(len(rets), 1)), index=rets.index, columns=["A"])

    engine_shift = Backtest(rets, CostModel(0.0)).run(w).gross_returns
    manual = (w.shift(1).fillna(0.0)["A"] * rets["A"])
    pd.testing.assert_series_equal(engine_shift, manual, check_names=False)


def test_costs_charged_on_turnover():
    """Buy-and-hold pays cost once (putting the book on); a flip-flop pays repeatedly."""
    rets = _returns()
    idx = rets.index

    hold = pd.DataFrame(1.0, index=idx, columns=["A"])
    cm = CostModel(per_side_bps=10.0)  # 0.001 per unit turnover
    res_hold = Backtest(rets, cm).run(hold)
    # Only the first day establishes the position -> turnover 1.0 that day, 0 after.
    assert np.isclose(res_hold.turnover.iloc[0], 1.0)
    assert np.isclose(res_hold.turnover.iloc[1:].sum(), 0.0)

    flip = pd.DataFrame({"A": [1.0, 0.0] * (len(idx) // 2)}, index=idx[: 2 * (len(idx) // 2)])
    flip = flip.reindex(idx).fillna(0.0)
    res_flip = Backtest(rets, cm).run(flip)
    assert res_flip.turnover.sum() > 10 * res_hold.turnover.sum()
