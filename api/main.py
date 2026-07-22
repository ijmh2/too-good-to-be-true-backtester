"""FastAPI backend for the too-good-to-be-true-backtester web UI.

Safe by construction: no user code is executed. A caller selects a strategy id and parameter
values from a fixed registry; the server runs the full overfit gauntlet and returns the
verdict, metrics, equity series and the composed scorecard PNG.

Local run:
    ./.venv/bin/uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.registry import REGISTRY, coerce, grid_of, public_schema
from api.serialize import scorecard_to_dict
from tgtbt.costs import CostModel
from tgtbt.data import get_prices, synthetic_prices
from tgtbt.reporting.scorecard import run_scorecard
from tgtbt.strategies import BuyAndHold

app = FastAPI(
    title="too-good-to-be-true-backtester API",
    description="Run the overfit gauntlet on built-in strategies. No user code executed.",
    version="1.0.0",
)

# Allow the Vercel frontend (and local dev) to call the API. Tighten to your domain in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    strategy_id: str
    params: dict[str, float] = Field(default_factory=dict)
    tickers: str = "SPY"
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    split: str = "2021-12-31"
    cost_bps: float = 5.0
    fast: bool = True
    n_folds: int = Field(default=5, ge=3, le=8)


@lru_cache(maxsize=64)
def _load_prices(tickers: tuple[str, ...], start: str, end: str) -> tuple:
    try:
        px = get_prices(list(tickers), start=start, end=end)
        if px.dropna(how="all").shape[0] > 250:
            return px, "live (yfinance)"
        raise RuntimeError("insufficient rows")
    except Exception as exc:  # noqa: BLE001 - graceful synthetic fallback
        n = max(750, (pd.Timestamp(end) - pd.Timestamp(start)).days)
        px = pd.concat(
            [synthetic_prices(t, n_days=n, seed=i) for i, t in enumerate(tickers)], axis=1
        )
        return px, f"synthetic fallback ({exc})"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/strategies")
def strategies() -> list[dict]:
    return public_schema()


@app.post("/run")
def run(req: RunRequest) -> dict:
    entry = REGISTRY.get(req.strategy_id)
    if entry is None:
        raise HTTPException(404, f"unknown strategy_id '{req.strategy_id}'")

    schema = entry["params"]
    params = coerce(schema, req.params)
    strategy = entry["factory"](**params)
    grid = grid_of(schema)

    tickers = tuple(t.strip().upper() for t in req.tickers.split(",") if t.strip())
    if not tickers:
        raise HTTPException(400, "no tickers provided")

    try:
        prices, source = _load_prices(tickers, req.start, req.end)
        cost_model = CostModel(req.cost_bps)
        benchmark = BuyAndHold().backtest(prices, cost_model=cost_model).net_returns
        n = 300 if req.fast else 1000
        card = run_scorecard(
            strategy, entry["factory"], grid, prices, benchmark=benchmark,
            split_date=req.split, cost_model=cost_model,
            perm_n=n, boot_n=n, n_folds=req.n_folds,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"run failed: {exc}") from exc

    payload = scorecard_to_dict(card)
    payload["data_source"] = source
    payload["params_used"] = params
    return payload
