"""Parsing for user-uploaded price data (CSV/Parquet) in the local Streamlit app.

Pure functions, no Streamlit dependency, so they're testable without a running app session.

Expected format: one date-like column (named `date`/`datetime`/`timestamp`/`time`, case
insensitive — or, failing that, the first column) plus one numeric column per asset. Column
headers become the asset names a strategy sees (e.g. via `prices.columns[0]`, or matched
against a `TICKERS`/`asset` a strategy script declares).
"""

from __future__ import annotations

import io

import pandas as pd

_DATE_ALIASES = {"date", "datetime", "timestamp", "time"}
MIN_ROWS = 30  # below this, most metrics/validation tools degrade to NaN anyway
RECOMMENDED_ROWS = 250  # walk-forward/CPCV are tuned assuming roughly a year+ of daily data


def parse_uploaded_prices(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse raw CSV/Parquet bytes into a wide (date index x asset columns) price frame.

    Returns the full parsed range, unsliced — callers filter to a start/end window themselves,
    the same way the yfinance path does.
    """
    buf = io.BytesIO(file_bytes)
    if filename.lower().endswith(".parquet"):
        df = pd.read_parquet(buf)
    else:
        df = pd.read_csv(buf)

    if df.shape[1] < 2:
        raise ValueError("need at least a date column and one price column")

    date_col = next(
        (c for c in df.columns if str(c).strip().lower() in _DATE_ALIASES), df.columns[0]
    )
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse '{date_col}' as dates: {exc}") from exc
    df = df.set_index(date_col).sort_index()
    df.index.name = "date"

    numeric = df.apply(pd.to_numeric, errors="coerce")
    dropped = [c for c in numeric.columns if numeric[c].isna().all()]
    numeric = numeric.drop(columns=dropped).dropna(how="all", axis=0)

    if numeric.shape[1] == 0:
        cols = ", ".join(f"'{c}'" for c in df.columns if c != date_col)
        raise ValueError(f"no numeric price columns found among: {cols}")
    if numeric.shape[0] < MIN_ROWS:
        raise ValueError(f"only {numeric.shape[0]} usable rows after parsing — need >= {MIN_ROWS}")
    if numeric.index.duplicated().any():
        raise ValueError("duplicate dates in the date column — de-duplicate before uploading")

    return numeric


def example_csv_bytes(n_days: int = 400) -> bytes:
    """A small synthetic two-asset CSV, for a 'download an example' button in the UI."""
    from tgtbt.data import synthetic_prices

    df = synthetic_prices(["AssetA", "AssetB"], n_days=n_days, seed=42)
    df.index.name = "Date"
    return df.reset_index().to_csv(index=False).encode()
