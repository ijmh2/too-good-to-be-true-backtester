"""Data layer: fetch daily prices, cache to parquet, plus a synthetic fallback.

We standardise on **adjusted close** (splits + dividends folded in) as the single price
series a strategy sees. Everything downstream works on a wide DataFrame indexed by date
with one column per ticker.

The synthetic generator exists so the engine and tests can run deterministically with no
network — good practice (tests shouldn't depend on Yahoo being up) and handy when the live
API is rate-limiting.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"


def _cache_path(tickers: list[str], start: str, end: str) -> Path:
    key = "|".join(sorted(tickers)) + f"|{start}|{end}"
    digest = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"prices_{digest}.parquet"


def get_prices(
    tickers: str | list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a wide DataFrame of adjusted close prices (rows=dates, cols=tickers).

    Results are cached to parquet keyed by (tickers, start, end); delete the cache file
    to force a refetch. Requires network on the first call for a given key.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(tickers, start, end)
    if use_cache and path.exists():
        return pd.read_parquet(path)

    import yfinance as yf  # imported lazily so offline/synthetic use needs no network

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
    )
    if raw.empty:
        raise RuntimeError(
            f"No data returned for {tickers} ({start}..{end}). "
            "Yahoo may be rate-limiting; retry, or use synthetic_prices() for testing."
        )
    # With auto_adjust=True, 'Close' is already the adjusted close.
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers)  # stable column order
    close = close.dropna(how="all").sort_index()

    close.to_parquet(path)
    return close


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a price frame. First row is dropped (undefined)."""
    return prices.pct_change().iloc[1:]


def synthetic_prices(
    tickers: str | list[str],
    n_days: int = 1000,
    *,
    seed: int = 0,
    mu: float = 0.08,
    sigma: float = 0.18,
    start: str = "2015-01-02",
) -> pd.DataFrame:
    """Deterministic geometric-Brownian-motion prices for offline testing.

    mu/sigma are annualised drift/vol. Every ticker gets an independent path from the
    same seed stream, so results are reproducible.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    dates = pd.bdate_range(start=start, periods=n_days)

    cols = {}
    for t in tickers:
        shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n_days)
        cols[t] = 100.0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(cols, index=dates)
