# Web — too-good-to-be-true-backtester UI

A Next.js (App Router) frontend for the overfit gauntlet. Pick a built-in strategy, set its
parameters, choose a universe/period/cost, and it renders the verdict, metric tiles, an
interactive equity curve, and the full scorecard figure — all from the FastAPI backend.

If the backend is running with `TGTBT_ALLOW_UPLOADS=1` (local dev only — see
[`../api/README.md`](../api/README.md)), the UI auto-detects it via `/config` and reveals two
extra options: upload your own strategy `.py` and/or your own price CSV. On a public
deployment (where that env var is never set) those options simply don't appear.

## Run locally

The backend must be running first (see [`../api/README.md`](../api/README.md)).

```bash
cp .env.local.example .env.local     # NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm install
npm run dev                          # http://localhost:3000
```

## Deploy on Vercel

1. Import the repo into Vercel.
2. Set **Root Directory** to `web`.
3. Framework preset is auto-detected as **Next.js** — no build overrides needed.
4. Add an environment variable **`NEXT_PUBLIC_API_URL`** = your deployed API URL
   (e.g. the Render service from `render.yaml`).
5. Deploy.

That's it — Vercel builds and hosts the static/SSR frontend; the heavy Python compute lives
on the separate API host.

## Stack

- Next.js 15 (App Router) + React 19, TypeScript
- Recharts for the interactive equity curve
- Palette mirrors the Python reporting style (validated CVD-safe hues), light + dark aware
