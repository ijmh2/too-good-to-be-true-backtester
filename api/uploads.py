"""Local-only upload support: custom strategy code and custom price data.

Both capabilities are gated behind `ALLOW_UPLOADS` in `api/main.py` (an env var, off by
default) because executing arbitrary Python is only ever safe when the caller is you, on your
own machine. Price-data parsing alone is not unsafe, but it's bundled under the same gate for
one simple mental model: "upload mode" is a local-dev feature, full stop.
"""

from __future__ import annotations

import io
import types

import pandas as pd

_DATE_ALIASES = {"date", "datetime", "timestamp", "time"}
MIN_ROWS = 30  # below this, most metrics/validation tools degrade to NaN anyway
RECOMMENDED_ROWS = 250  # walk-forward/CPCV are tuned assuming roughly a year+ of daily data
MAX_UPLOAD_CHARS = 2_000_000  # ~2MB of text; a soft guard against pathological payloads


def parse_uploaded_prices(csv_text: str) -> pd.DataFrame:
    """Parse raw CSV text into a wide (date index x asset columns) price frame.

    Expected shape: a date-like column (`date`/`datetime`/`timestamp`/`time`, case
    insensitive — or, failing that, the first column) plus one numeric column per asset.
    Column headers become the asset names a strategy sees. Returns the full parsed range,
    unsliced — callers filter to a start/end window themselves.
    """
    if len(csv_text) > MAX_UPLOAD_CHARS:
        raise ValueError(f"file too large ({len(csv_text)} chars, max {MAX_UPLOAD_CHARS})")

    df = pd.read_csv(io.StringIO(csv_text))
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


def example_csv_text(n_days: int = 400) -> str:
    """A small synthetic two-asset CSV, for a 'download an example' link in the UI."""
    from tgtbt.data import synthetic_prices

    df = synthetic_prices(["AssetA", "AssetB"], n_days=n_days, seed=42)
    df.index.name = "Date"
    return df.reset_index().to_csv(index=False)


def exec_strategy_code(code: str) -> types.ModuleType:
    """Execute uploaded strategy source in a fresh module namespace.

    Requires module-level `STRATEGY`, `FACTORY`, `GRID` — see `app/strategy_template.py` for
    the contract and a worked example. Deliberately unsandboxed: this is a local research tool,
    same trust model as running a notebook cell.
    """
    if len(code) > MAX_UPLOAD_CHARS:
        raise ValueError(f"script too large ({len(code)} chars, max {MAX_UPLOAD_CHARS})")

    mod = types.ModuleType("user_strategy")
    try:
        exec(compile(code, "<uploaded strategy>", "exec"), mod.__dict__)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"strategy code failed to execute: {exc}") from exc

    for attr in ("STRATEGY", "FACTORY", "GRID"):
        if not hasattr(mod, attr):
            raise ValueError(f"uploaded script must define a module-level `{attr}`")
    return mod
