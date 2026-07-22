# API — too-good-to-be-true-backtester backend

A FastAPI service that runs the overfit gauntlet. Two modes:

- **Public mode (default).** A caller picks a `strategy_id` and parameter *values* from a
  fixed registry, plus a ticker universe. No user code is ever executed — safe to expose
  publicly.
- **Local upload mode** (`TGTBT_ALLOW_UPLOADS=1`). Additionally accepts raw strategy source
  (`strategy_code`) and/or raw CSV price data (`price_csv`) in the request body. This executes
  arbitrary Python — **only enable it when you are the only one who can reach this server**
  (e.g. running on `localhost` for your own research). It is off by default and the Docker
  image / `render.yaml` never set it, so the public deployment can't be tricked into turning
  it on.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe. |
| `GET` | `/config` | `{"allow_uploads": bool}` — lets the frontend feature-detect upload mode. |
| `GET` | `/strategies` | Catalogue: built-in strategy ids, descriptions, parameter schemas. |
| `GET` | `/example-csv` | A small synthetic CSV, for the frontend's "download an example" link. |
| `POST` | `/run` | Run the gauntlet; returns verdict, metrics, equity series, and the scorecard PNG. |

`POST /run` body — built-in strategy + tickers (always available):

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

Or, with `TGTBT_ALLOW_UPLOADS=1` — your own code + your own data (either or both, alongside or
instead of `strategy_id`/`tickers`):

```json
{
  "strategy_code": "STRATEGY = ...\nFACTORY = ...\nGRID = ...",
  "price_csv": "date,AssetA,AssetB\n2020-01-01,100,50\n...",
  "start": "2020-01-01", "end": "2024-12-31", "split": "2023-06-30"
}
```

`strategy_code` must define module-level `STRATEGY`, `FACTORY`, `GRID` — see
[`../app/strategy_template.py`](../app/strategy_template.py) for the contract. Requests using
`strategy_code` or `price_csv` return `403` if `ALLOW_UPLOADS` is off.

## Run locally

```bash
pip install -e ".[api]"          # from repo root; installs tgtbt + fastapi/uvicorn
uvicorn api.main:app --reload --port 8000
# docs at http://127.0.0.1:8000/docs

# to also enable local upload mode:
TGTBT_ALLOW_UPLOADS=1 uvicorn api.main:app --reload --port 8000
```

## Deploy (Docker)

The repo-root `Dockerfile` builds the image; `render.yaml` gives a one-click Render deploy.
Any Docker host works (Render / Railway / Fly.io / Cloud Run):

```bash
docker build -t tgtbt-api .
docker run -p 8000:8000 tgtbt-api   # ALLOW_UPLOADS unset -> public-safe mode
```

The host injects `$PORT`. After deploying, set the frontend's `NEXT_PUBLIC_API_URL` to the
service URL. CORS is currently open (`*`); tighten `allow_origins` in `api/main.py` to your
Vercel domain for production. **Never set `TGTBT_ALLOW_UPLOADS` on a publicly reachable
deployment.**

**Note on `/run` cost:** each call runs walk-forward + permutation + Monte-Carlo + CSCV
(seconds of CPU). There is no auth or rate limiting — fine for a personal/portfolio deploy;
add a limiter before exposing it widely.
