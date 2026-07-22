"""FastAPI backend for the too-good-to-be-true-backtester web UI.

Two modes, one codebase:

- **Public mode (default).** A caller selects a built-in `strategy_id` and parameter values
  from a fixed registry, and a ticker universe. No user code is ever executed. This is what
  runs on the public deploy (Vercel frontend + this API on Render/etc).
- **Local upload mode.** Set `TGTBT_ALLOW_UPLOADS=1` to additionally accept a raw strategy
  script (`strategy_code`) and/or raw CSV price data (`price_csv`) in the request body. This
  executes arbitrary Python — only ever enable it when you are the only one who can reach this
  server (e.g. running `uvicorn` on localhost for your own research). It is OFF by default and
  the Docker/Render deploy config never sets it, so the public instance can't be tricked into
  turning it on.

Local run (public-safe defaults):
    ./.venv/bin/uvicorn api.main:app --reload --port 8000

Local run with uploads enabled (only do this on your own machine):
    TGTBT_ALLOW_UPLOADS=1 ./.venv/bin/uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.registry import REGISTRY, coerce, grid_of, public_schema
from api.serialize import scorecard_to_dict
from api.uploads import example_csv_text, exec_strategy_code, parse_uploaded_prices
from tgtbt.costs import CostModel
from tgtbt.data import get_prices, synthetic_prices
from tgtbt.reporting.scorecard import run_scorecard
from tgtbt.strategies import BuyAndHold

# Off by default. Only set locally — never in the Docker image / Render config — since this
# flag is what makes exec()-on-request possible.
ALLOW_UPLOADS = os.environ.get("TGTBT_ALLOW_UPLOADS", "0").strip().lower() in ("1", "true", "yes")

app = FastAPI(
    title="too-good-to-be-true-backtester API",
    description="Run the overfit gauntlet on built-in strategies (and, in local upload mode, "
    "your own strategy code and price data).",
    version="1.1.0",
)

# Allow the Vercel frontend (and local dev) to call the API. Tighten to your domain in prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    # Built-in path (always available).
    strategy_id: str | None = None
    params: dict[str, float] = Field(default_factory=dict)
    tickers: str = "SPY"

    # Upload path — requires ALLOW_UPLOADS=1 on the server, enforced in the handler below.
    strategy_code: str | None = None  # raw .py source defining STRATEGY/FACTORY/GRID
    price_csv: str | None = None      # raw CSV text: date column + one numeric col per asset

    start: str = "2010-01-01"
    end: str = "2024-12-31"
    split: str = "2021-12-31"
    cost_bps: float = 5.0
    fast: bool = True
    n_folds: int = Field(default=5, ge=3, le=8)


@lru_cache(maxsize=64)
def _load_ticker_prices(tickers: tuple[str, ...], start: str, end: str) -> tuple:
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


@app.get("/config")
def config() -> dict:
    """Frontend feature-detection: is this deployment allowed to run uploaded code/data?"""
    return {"allow_uploads": ALLOW_UPLOADS}


@app.get("/strategies")
def strategies() -> list[dict]:
    return public_schema()


@app.get("/example-csv")
def example_csv() -> dict:
    """A small synthetic CSV for the frontend's 'download an example' link (upload mode)."""
    return {"filename": "example_prices.csv", "content": example_csv_text()}


@app.post("/run")
def run(req: RunRequest) -> dict:
    # --- resolve the strategy: built-in registry, or uploaded code (gated) ---
    if req.strategy_code is not None:
        if not ALLOW_UPLOADS:
            raise HTTPException(403, "strategy upload is disabled on this deployment")
        try:
            mod = exec_strategy_code(req.strategy_code)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        strategy, factory, grid = mod.STRATEGY, mod.FACTORY, mod.GRID
        params_used: dict = {}
    elif req.strategy_id is not None:
        entry = REGISTRY.get(req.strategy_id)
        if entry is None:
            raise HTTPException(404, f"unknown strategy_id '{req.strategy_id}'")
        schema = entry["params"]
        params_used = coerce(schema, req.params)
        strategy = entry["factory"](**params_used)
        factory = entry["factory"]
        grid = grid_of(schema)
    else:
        raise HTTPException(400, "provide either strategy_id or strategy_code")

    # --- resolve the price data: tickers, or uploaded CSV (gated) ---
    if req.price_csv is not None:
        if not ALLOW_UPLOADS:
            raise HTTPException(403, "data upload is disabled on this deployment")
        try:
            full = parse_uploaded_prices(req.price_csv)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        prices = full.loc[pd.Timestamp(req.start) : pd.Timestamp(req.end)]
        if prices.shape[0] < 30:
            raise HTTPException(
                400, f"only {prices.shape[0]} rows between {req.start} and {req.end} — "
                     "widen the date range"
            )
        source = "uploaded CSV"
    else:
        tickers = tuple(t.strip().upper() for t in req.tickers.split(",") if t.strip())
        if not tickers:
            raise HTTPException(400, "no tickers provided")
        prices, source = _load_ticker_prices(tickers, req.start, req.end)

    try:
        cost_model = CostModel(req.cost_bps)
        benchmark = BuyAndHold().backtest(prices, cost_model=cost_model).net_returns
        n = 300 if req.fast else 1000
        card = run_scorecard(
            strategy, factory, grid, prices, benchmark=benchmark,
            split_date=req.split, cost_model=cost_model,
            perm_n=n, boot_n=n, n_folds=req.n_folds,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"run failed: {exc}") from exc

    payload = scorecard_to_dict(card)
    payload["data_source"] = source
    payload["params_used"] = params_used
    return payload
