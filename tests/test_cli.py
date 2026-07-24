"""CLI smoke tests.

Exercised in-process (calling `main()` directly) for speed, with a monkeypatched
deterministic price source so `run`/`batch` need no network. The actual end-user experience —
the real `tgtbt` console script against live Yahoo data — was verified by hand separately;
these guard against regressions in the argument parsing / wiring going forward.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tgtbt.cli import main
from tgtbt.data import synthetic_prices


def _fake_prices(tickers, start=None, end=None, **_kwargs):
    """Matches both call shapes: cli.py passes a list, batch.py passes one ticker at a time."""
    ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)
    frames = [
        synthetic_prices(t, n_days=600, seed=sum(ord(c) for c in t)) for t in ticker_list
    ]
    return pd.concat(frames, axis=1), "synthetic (test)"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr("tgtbt.cli.get_prices_or_fallback", _fake_prices)
    monkeypatch.setattr("tgtbt.batch.get_prices_or_fallback", _fake_prices)


def test_list_runs(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for sid in ("trend", "meanrev", "dualmom", "bollinger"):
        assert sid in out


def test_run_missing_strategy():
    with pytest.raises(SystemExit, match="provide --strategy"):
        main(["run", "--ticker", "SPY"])


def test_run_unknown_strategy():
    with pytest.raises(SystemExit, match="unknown --strategy"):
        main(["run", "--strategy", "nope", "--ticker", "SPY"])


def test_run_builtin(tmp_path, capsys):
    out_png = tmp_path / "out.png"
    code = main(
        ["run", "--strategy", "trend", "--ticker", "FAKE", "--folds", "3", "--out", str(out_png)]
    )
    assert code == 0
    assert out_png.exists()
    captured = capsys.readouterr().out
    assert "VERDICT" in captured
    assert "data: synthetic (test)" in captured


def test_run_custom_strategy_file(tmp_path, capsys):
    strategy_file = tmp_path / "my_strategy.py"
    strategy_file.write_text(
        """
import pandas as pd
from tgtbt.strategies.base import Strategy

class MyStrat(Strategy):
    def __init__(self, window: int = 50):
        self.window = window
        self.name = f"my({window})"
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        px = prices.iloc[:, 0]
        ma = px.rolling(self.window, min_periods=self.window).mean()
        w = (px > ma).astype(float).fillna(0.0)
        return w.to_frame(prices.columns[0]).reindex(columns=prices.columns).fillna(0.0)

def make(**p):
    return MyStrat(**p)

STRATEGY = MyStrat(window=50)
FACTORY = make
GRID = {"window": [20, 50, 100]}
"""
    )
    out_png = tmp_path / "out.png"
    code = main(
        [
            "run", "--strategy-file", str(strategy_file), "--ticker", "FAKE",
            "--param", "window=30", "--folds", "3", "--out", str(out_png),
        ]
    )
    assert code == 0
    assert out_png.exists()
    assert "my(30)" in capsys.readouterr().out  # confirms --param actually overrode the default


def test_run_price_csv(tmp_path, capsys):
    from tgtbt.loaders import example_csv_text

    csv_file = tmp_path / "my_data.csv"
    csv_file.write_text(example_csv_text(n_days=400))
    out_png = tmp_path / "out.png"
    code = main(
        [
            "run", "--strategy", "trend", "--price-csv", str(csv_file),
            "--start", "2015-01-01", "--end", "2016-12-31", "--folds", "3",
            "--out", str(out_png),
        ]
    )
    assert code == 0
    assert "uploaded CSV" in capsys.readouterr().out


def test_batch(tmp_path, capsys):
    csv_out = tmp_path / "batch.csv"
    code = main(
        [
            "batch", "--strategy", "bollinger", "--tickers", "FAKE_A,FAKE_B",
            "--folds", "3", "--csv-out", str(csv_out),
        ]
    )
    assert code == 0
    assert csv_out.exists()
    out = capsys.readouterr().out
    assert "FAKE_A" in out and "FAKE_B" in out
