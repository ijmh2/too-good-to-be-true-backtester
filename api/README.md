# API — too-good-to-be-true-backtester backend

A small FastAPI service that runs the overfit gauntlet on **built-in** strategies. It executes
**no user-supplied code** — a caller picks a `strategy_id` and parameter *values* from a fixed
registry — so it is safe to expose publicly.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe. |
| `GET` | `/strategies` | Catalogue: strategy ids, descriptions, and parameter schemas. |
| `POST` | `/run` | Run the gauntlet; returns verdict, metrics, equity series, and the scorecard PNG. |

`POST /run` body:

```json
{
  "strategy_id": "trend",
  "params": { "trend_window": 200, "vol_window": 20, "target_vol": 0.15 },
  "tickers": "SPY",
  "start": "2010-01-01",
  "end": "2024-12-31",
  "split": "2021-12-31",
  "cost_bps": 5,
  "fast": true,
  "n_folds": 5
}
```

## Run locally

```bash
pip install -e ".[ui]"          # from repo root; installs the tgtbt package
pip install fastapi "uvicorn[standard]"
uvicorn api.main:app --reload --port 8000
# docs at http://127.0.0.1:8000/docs
```

## Deploy (Docker)

The repo-root `Dockerfile` builds the image; `render.yaml` gives a one-click Render deploy.
Any Docker host works (Render / Railway / Fly.io / Cloud Run):

```bash
docker build -t tgtbt-api .
docker run -p 8000:8000 tgtbt-api
```

The host injects `$PORT`. After deploying, set the frontend's `NEXT_PUBLIC_API_URL` to the
service URL. CORS is currently open (`*`); tighten `allow_origins` in `api/main.py` to your
Vercel domain for production.

**Note on `/run` cost:** each call runs walk-forward + permutation + Monte-Carlo + CSCV
(seconds of CPU). There is no auth or rate limiting — fine for a personal/portfolio deploy;
add a limiter before exposing it widely.
