"""Built-in strategy registry with parameter schemas for the public API.

The public API deliberately never executes user-supplied code — a caller picks a strategy id
and parameter *values* from this fixed catalogue. Each param carries UI hints (type, range,
default) plus a `grid`: the search space the validation gauntlet stress-tests over. The
headline strategy uses the caller's chosen values; the grid is what walk-forward / CPCV /
deflated-Sharpe explore.
"""

from __future__ import annotations

from tgtbt.strategies import make_dual_momentum, make_mean_reversion, make_trend_vt

REGISTRY: dict[str, dict] = {
    "trend": {
        "name": "Trend + volatility target",
        "description": "Hold the asset only above its moving average, scaled to a volatility "
                       "target. Cuts drawdowns; academically well-supported.",
        "factory": make_trend_vt,
        "default_tickers": "SPY",
        "params": [
            {"name": "trend_window", "label": "Trend MA window (days)", "type": "int",
             "default": 200, "min": 20, "max": 300, "step": 10, "grid": [50, 100, 150, 200]},
            {"name": "vol_window", "label": "Volatility lookback (days)", "type": "int",
             "default": 20, "min": 5, "max": 90, "step": 5, "grid": [20, 40, 60]},
            {"name": "target_vol", "label": "Target annual volatility", "type": "float",
             "default": 0.15, "min": 0.05, "max": 0.40, "step": 0.01, "grid": [0.15]},
        ],
    },
    "meanrev": {
        "name": "Short-horizon mean reversion",
        "description": "Buy the dip: go long when price is stretched below its short moving "
                       "average. A classic in-sample-pretty, cost-fragile trap.",
        "factory": make_mean_reversion,
        "default_tickers": "SPY",
        "params": [
            {"name": "lookback", "label": "Lookback window (days)", "type": "int",
             "default": 5, "min": 2, "max": 30, "step": 1, "grid": [3, 5, 10, 20]},
            {"name": "entry_z", "label": "Entry z-score (below)", "type": "float",
             "default": -1.0, "min": -3.0, "max": -0.25, "step": 0.25,
             "grid": [-0.5, -1.0, -1.5, -2.0]},
        ],
    },
    "dualmom": {
        "name": "Dual momentum (rotation)",
        "description": "Rotate into the strongest asset by trailing return, but only if that "
                       "return is positive; otherwise sit in cash.",
        "factory": make_dual_momentum,
        "default_tickers": "SPY,TLT,EFA",
        "params": [
            {"name": "lookback", "label": "Momentum lookback (days)", "type": "int",
             "default": 126, "min": 20, "max": 300, "step": 10, "grid": [63, 126, 252]},
        ],
    },
}


def coerce(param_schema: list[dict], values: dict) -> dict:
    """Coerce, clamp, and whitelist caller-supplied values against the schema.

    Unknown keys are ignored; missing keys fall back to the default; every value is clamped to
    the schema's [min, max] so an out-of-range param can't reach (and crash) the backtest.
    """
    out = {}
    for p in param_schema:
        v = values.get(p["name"], p["default"])
        v = int(v) if p["type"] == "int" else float(v)
        v = max(p["min"], min(p["max"], v))  # clamp to declared bounds
        out[p["name"]] = int(v) if p["type"] == "int" else v
    return out


def grid_of(param_schema: list[dict]) -> dict[str, list]:
    return {p["name"]: p["grid"] for p in param_schema}


def public_schema() -> list[dict]:
    """Registry as JSON-safe metadata (no callables) for GET /strategies."""
    return [
        {"id": sid, "name": e["name"], "description": e["description"],
         "default_tickers": e["default_tickers"], "params": e["params"]}
        for sid, e in REGISTRY.items()
    ]
