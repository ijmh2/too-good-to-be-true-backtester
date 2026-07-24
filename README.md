# too-good-to-be-true-backtester

> If a backtest looks too good to be true, this framework tells you *why*.

A strategy-validation harness built around one uncomfortable fact: **most backtested
"edges" are overfitting artefacts.** You hand it a trading strategy; it runs the full
gauntlet — out-of-sample splits, walk-forward re-fitting, parameter-robustness surfaces,
permutation / random-timing nulls, Monte-Carlo confidence cones, the Deflated Sharpe
(multiple-testing correction), and CSCV → probability of backtest overfitting — then returns
a plain-English **verdict**: *likely real edge, inconclusive, or likely overfit.*

Look-ahead is prevented structurally (a strategy physically never sees future bars);
transaction costs are on from the first backtest.

## Quick start — one command, full depth

```bash
git clone https://github.com/ijmh2/too-good-to-be-true-backtester.git
cd too-good-to-be-true-backtester
python -m venv .venv && source .venv/bin/activate
pip install -e .

tgtbt list                                    # see the built-in strategies
tgtbt run --strategy trend --ticker SPY       # runs the ENTIRE gauntlet, prints the verdict,
                                               # saves a scorecard PNG — no server, no Node.js
```

That's it. No web server, no Node/npm, no notebook. Every statistical check described above
runs on every `tgtbt run` — there is no "lite" mode; simple-to-use and in-depth are the same
command.

```
$ tgtbt run --strategy trend --ticker SPY
data: live (yfinance), 3773 rows, 2010-01-04 -> 2024-12-30
running the gauntlet: 45 configs x 5-fold walk-forward, 300 permutation resamples, ...

============================================================
VERDICT: INCONCLUSIVE   (0.9s)
============================================================
  PASS  oos_positive
  FAIL  timing_significant
  PASS  beats_deflated_bar
  PASS  low_overfit_prob

| Metric                 | Value |
|-------------------------|-------|
| Full-sample Sharpe      | 0.89  |
| Walk-forward OOS Sharpe | 0.85  |
| Permutation p-value     | 0.070 |
| Deflated Sharpe (DSR)   | 0.99  |
| Prob. backtest overfit  | 0.25  |
...
[saved] scorecard_trend_vt.png
```

### Test your own strategy — still one command

```bash
tgtbt run --strategy-file my_strategy.py --ticker AAPL
```

`my_strategy.py` just needs three module-level names — `STRATEGY`, `FACTORY`, `GRID` — see
[`app/strategy_template.py`](app/strategy_template.py) for the ~30-line contract and a working
example to copy. No web upload, no code review gate: it's your machine, your file, executed
directly — same trust model as running a Python script (because it is one).

### Test your own data

```bash
tgtbt run --strategy trend --price-csv my_prices.csv
```

Any CSV with a date column and one numeric column per asset (headers become the asset names)
— no Yahoo Finance account or API key needed.

### Does the edge survive on other tickers?

```bash
tgtbt batch --strategy bollinger --tickers TSLA,NVDA,AAPL,SPY,QQQ
```

Runs the identical strategy/grid across every ticker and prints one sorted comparison table
(`likely real edge` → `inconclusive` → `likely overfit`), instead of you re-running `tgtbt run`
by hand per ticker.

### More history, or more/fewer resamples

```bash
tgtbt run --strategy trend --ticker SPY --max-history     # fetches everything Yahoo has
tgtbt run --strategy trend --ticker SPY --thorough         # 1000 resamples instead of 300
```

Run `tgtbt <list|run|batch> --help` for the full flag reference.

## Why this exists

Anyone can produce a beautiful equity curve by tuning parameters until the past looks
profitable. The hard — and genuinely valuable — skill is telling a real signal apart from a
lucky fit. This repo is a toolkit for exactly that question.

**New to terms like Sharpe, Deflated Sharpe, or PBO?** See [`GLOSSARY.md`](GLOSSARY.md) —
every metric in the output, defined precisely to match what the code computes. The web UI
also has an inline "What do these mean?" panel next to the results.

## Example output

Same framework, two strategies on SPY (2010–2024, in-sample ≤ 2021, out-of-sample 2022+):

![Trend scorecard](docs/scorecard_trend.png)

**Trend + volatility targeting → `INCONCLUSIVE`.** Sharpe 0.89, walk-forward out-of-sample
Sharpe 0.91, low overfit probability (PBO 0.12) — but the permutation p-value is 0.088, so the
*timing* isn't quite distinguishable from a random-timing strategy with the same exposure, and
the harness declines to call it a real edge.

**Short-horizon mean reversion → `LIKELY OVERFIT`.** A pretty in-sample curve, but PBO 0.61
and a deflated Sharpe of 0.79 — the search is more likely fitting noise than finding signal.
(See [`docs/scorecard_meanrev.png`](docs/scorecard_meanrev.png).)

That an honestly-built framework returns "inconclusive / overfit" rather than a fantasy Sharpe
is the entire point. See also `tgtbt batch --strategy bollinger --tickers TSLA,NVDA,AAPL,SPY,QQQ`:
the same breakout strategy shows a real edge on TSLA and AAPL, is inconclusive on the
diversified SPY/QQQ, and is flagged *likely overfit* on NVDA specifically because its overfit
probability is 0.81 despite a superficially decent Sharpe of 0.91 — the deeper diagnostics
catching what a Sharpe-only check would miss.

## Design principles

1. **No look-ahead, by construction.** The engine controls the clock; a strategy is only
   ever handed data up to time *t* and its weights are applied at *t+1*. Leakage isn't
   discouraged — it's made impossible by the interface.
2. **Costs from the start.** Turnover-based transaction costs are applied in every backtest,
   because an event-driven edge that ignores costs isn't an edge.
3. **Assume you're fooling yourself.** Every result is checked against a null (random timing /
   permuted signals) and corrected for the number of trials.
4. **"Real edge" ≠ "beats buy-and-hold".** The verdict certifies a timing signal is
   statistically genuine — not that the strategy out-earns a fully-invested benchmark, which a
   lower-beta strategy usually won't in a rising market. Alpha and beta vs. the benchmark are
   always reported alongside Sharpe so you can tell the two questions apart.

## Writing your own strategy

Subclass `Strategy` and implement one method, using only backward-looking windows:

```python
class MyStrategy(Strategy):
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        ...  # weights.loc[t] may depend only on prices up to and including row t
```

Then either point the CLI at it (`tgtbt run --strategy-file my_strategy.py --ticker ...`), or
drive it from Python directly:

```python
from tgtbt.data import get_prices_or_fallback
from tgtbt.strategies import BuyAndHold
from tgtbt.reporting.scorecard import run_scorecard
from my_strategy import STRATEGY, FACTORY, GRID

prices, source = get_prices_or_fallback("SPY", start="2010-01-01", end="2024-12-31")
benchmark = BuyAndHold().backtest(prices).net_returns
card = run_scorecard(STRATEGY, FACTORY, GRID, prices, benchmark=benchmark, split_date="2021-12-31")
print(card.verdict)          # 'likely real edge' | 'inconclusive' | 'likely overfit'
card.figure().savefig("scorecard.png")
```

## Optional: web UI

A Next.js frontend over a FastAPI backend, for a nicer visual experience or a shareable public
link — **not required** for any of the above, and needs Node.js in addition to Python:

```bash
pip install -e ".[api]"
uvicorn api.main:app --port 8000            # terminal 1 (backend)
cd web && npm install && npm run dev        # terminal 2 (frontend -> http://localhost:3000)
```

By default this is the same "no user code executed" public mode the CLI's `--strategy`
built-ins use. Set `TGTBT_ALLOW_UPLOADS=1` to also unlock pasting your own strategy code/CSV
data in the browser (mirrors `--strategy-file`/`--price-csv` above) — **only do this on your
own machine**; it's off by default and never set in the Docker image or `render.yaml`, so a
public deployment can't be tricked into running arbitrary code. Deploy: `web/` on Vercel,
`api/` on any Docker host (`Dockerfile` + `render.yaml` included). Details in
[`api/README.md`](api/README.md) and [`web/README.md`](web/README.md).

## Layout

```
tgtbt/
  cli.py             # `tgtbt` console command: list / run / batch
  registry.py         # built-in strategy catalogue (shared by the CLI, API, and web UI)
  loaders.py          # load a custom strategy file / custom price CSV
  batch.py            # run one strategy across many tickers, one sorted comparison table
  data.py              # price fetch (yfinance) + parquet cache + synthetic fallback
  costs.py             # turnover-based transaction-cost model
  engine.py            # causal weights -> portfolio-returns backtester
  metrics.py           # CAGR, Sharpe, Sortino, max drawdown, Calmar, alpha/beta
  strategies/          # Strategy base + buy&hold, trend/vol-target, mean-reversion,
                       #   dual-momentum, Bollinger breakout
  validation/          # walk-forward, robustness surface, permutation, Monte-Carlo,
                       #   deflated Sharpe (PSR/DSR), CSCV -> PBO
  reporting/           # chart builders + the composed overfit scorecard
app/                   # strategy_template.py (the STRATEGY/FACTORY/GRID contract) + AGENT_PROMPT.md
api/, web/             # optional web UI (see "Optional: web UI" above)
examples/              # runnable end-to-end scripts
tests/                 # look-ahead leak test, engine correctness, CLI + validation-stat checks
docs/                  # committed example scorecard figures
```

## Caveats (read these)

- **Daily data, vectorised engine.** Not an order-book simulator — appropriate for
  daily-rebalanced strategies, not HFT.
- **Survivorship bias.** `yfinance` returns today's listings; prefer liquid ETFs and treat
  single-name universes with caution.
- This is a **research and educational** tool for judging strategies, not investment advice.
