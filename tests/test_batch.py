"""Tests for batch multi-ticker testing.

Uses a monkeypatched, deterministic price source (no network) so the test is reproducible and
fast; the point here is structural correctness of the batch runner, not any particular
strategy's real-world verdict.
"""

from __future__ import annotations

import pandas as pd

from tgtbt.batch import run_batch, _VERDICT_ORDER
from tgtbt.data import synthetic_prices
from tgtbt.strategies.trend import make_trend_vt


def _deterministic_seed(ticker: str) -> int:
    return sum(ord(c) for c in ticker)


def _fake_prices(ticker, start=None, end=None, **_kwargs):
    return synthetic_prices(ticker, n_days=800, seed=_deterministic_seed(ticker)), "synthetic (test)"


def test_run_batch_structure_and_sorting(monkeypatch):
    monkeypatch.setattr("tgtbt.batch.get_prices_or_fallback", _fake_prices)

    tickers = ["FAKE_A", "FAKE_B", "FAKE_C"]
    df = run_batch(
        make_trend_vt,
        headline_params={"trend_window": 100, "vol_window": 20, "target_vol": 0.15},
        grid={"trend_window": [50, 100], "vol_window": [20], "target_vol": [0.15]},
        tickers=tickers,
        perm_n=50,
        boot_n=50,
        n_folds=3,
        verbose=False,
    )

    assert set(df.index) == set(tickers)
    expected_cols = {
        "verdict", "data_source", "split_date", "sharpe", "wf_oos_sharpe",
        "perm_p", "dsr", "pbo", "max_drawdown",
    }
    assert expected_cols <= set(df.columns)
    assert df["verdict"].isin(_VERDICT_ORDER).all()
    assert (df["data_source"] == "synthetic (test)").all()

    # rows must be sorted "likely real edge" -> "inconclusive" -> "likely overfit"
    ranks = df["verdict"].map(_VERDICT_ORDER).tolist()
    assert ranks == sorted(ranks)


def test_run_batch_is_reproducible(monkeypatch):
    monkeypatch.setattr("tgtbt.batch.get_prices_or_fallback", _fake_prices)

    kwargs = dict(
        factory=make_trend_vt,
        headline_params={"trend_window": 100, "vol_window": 20, "target_vol": 0.15},
        grid={"trend_window": [50, 100], "vol_window": [20], "target_vol": [0.15]},
        tickers=["FAKE_A"],
        perm_n=50,
        boot_n=50,
        n_folds=3,
        verbose=False,
    )
    df1 = run_batch(**kwargs)
    df2 = run_batch(**kwargs)
    pd.testing.assert_frame_equal(df1, df2)
