# Prompt: generate a new strategy for too-good-to-be-true-backtester

Copy everything below the line into a fresh agent/session (it's self-contained — the agent
doesn't need any other context from this repo's history). Fill in the bracketed idea before
sending.

---

You are writing a trading strategy for **too-good-to-be-true-backtester**, a Python
strategy-validation harness whose whole thesis is *detecting overfitting*, not finding real
alpha. The repo is at `~/projects/too-good-to-be-true-backtester` with a venv at `.venv`
(`source .venv/bin/activate` before running anything). Do not commit or push — just write the
file and prove it runs; the human will review before committing.

## The idea to implement

[DESCRIBE THE STRATEGY IDEA HERE — e.g. "a Kalman-filter pairs trade on two cointegrated ETFs
with a time-varying hedge ratio" or "a cross-sectional momentum rotation across 8 sector ETFs"
or "a volatility-regime filter using a rolling z-score of realized vol". Include any economic
rationale — this harness cares about *why* an edge might exist, not just curve-fitting.]

## The contract you must satisfy

A strategy is a subclass of `tgtbt.strategies.base.Strategy` implementing exactly one method:

```python
def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
    ...  # return target weights, same index/columns as `prices` (or a subset of columns)
```

**Hard rule — no look-ahead:** `weights.loc[t]` may depend only on `prices` up to and
including row `t`. Every rolling/expanding window must be **backward-looking**
(`.rolling(window)`, never centered or shifted forward). The engine applies your weights with
a `shift(1)` before multiplying by returns, so a decision at `t` is only ever paid the return
realised strictly after `t` — but that only helps you if your *indicators* are also strictly
backward-looking. Look at `tgtbt/strategies/trend.py` and `tgtbt/strategies/mean_reversion.py`
for two working examples of this pattern.

Weights should be in `[-1, 1]` per asset for a single long/short name, or sum to `<= 1` across
assets for a long-only cross-sectional/rotation strategy (see
`tgtbt/strategies/dual_momentum.py` and the `equal_weight()` helper in
`tgtbt/strategies/base.py` if you need to split an allocation across several selected assets).
Fill NaNs (warm-up periods) with `0.0` — that means "flat / in cash", which is always safe.

## What to produce

1. A new file `tgtbt/strategies/<name>.py` containing:
   - The `Strategy` subclass (with a descriptive `self.name` set in `__init__` reflecting its
     parameters, e.g. `f"my_strategy(window={window})"`).
   - A `make_<name>(**params) -> <Class>` factory function (this lets the validation tools —
     walk-forward, parameter surface, CPCV — re-parameterise your strategy automatically).
2. Register it in `tgtbt/strategies/__init__.py` (import + add to `__all__`), following the
   existing pattern for the other strategies in that file.
3. A test in `tests/test_validation.py` (or a new `tests/test_<name>.py`) that at minimum
   checks: no NaNs in the output weights, weights are bounded/normalised as above, and the
   warm-up period is correctly flat (zero) before your longest lookback window is satisfied.
   Follow the style of `test_trend_weights_are_valid_and_bounded` in `tests/test_validation.py`.
4. Run the full suite (`pytest -q` from repo root) and confirm everything passes, including
   your new test.
5. Run your strategy through the actual scorecard on real data as a smoke test — easiest via
   the CLI once the strategy is registered in `tgtbt/strategies/__init__.py`:

   ```bash
   tgtbt run --strategy-file /tmp/<name>_standalone.py --ticker SPY --out /tmp/<name>_scorecard.png
   ```

   (that needs a standalone copy with module-level `STRATEGY`/`FACTORY`/`GRID` appended — see
   `app/strategy_template.py` — separate from the repo-integrated file, which shouldn't define
   those globals; see the other files in `tgtbt/strategies/` for the pattern). Or from Python
   directly:

```python
from tgtbt.data import get_prices_or_fallback
from tgtbt.strategies import BuyAndHold
from tgtbt.reporting.scorecard import run_scorecard
from tgtbt.strategies.<name> import <Class>, make_<name>

prices, source = get_prices_or_fallback("SPY", start="2010-01-01", end="2024-12-31")
benchmark = BuyAndHold().backtest(prices).net_returns
grid = {...}  # 1-3 parameters, 2-4 values each — this is what walk-forward/CPCV sweep over
card = run_scorecard(<Class>(...), make_<name>, grid, prices, benchmark=benchmark,
                     split_date="2021-12-31", perm_n=200, boot_n=200, n_folds=4)
print(card.verdict, card.flags)
card.figure().savefig("/tmp/<name>_scorecard.png")
```

`get_prices_or_fallback` already falls back to `synthetic_prices` automatically if Yahoo is
unreachable/rate-limited — this is a normal offline fallback used throughout the repo, not a
workaround you need to justify.

## What "good" looks like

Report back: the strategy's verdict (`likely real edge` / `inconclusive` / `likely overfit`),
which of the four gates it passed (`oos_positive`, `timing_significant`,
`beats_deflated_bar`, `low_overfit_prob`), and your honest read on *why* — e.g. "the parameter
surface is a lonely spike, this is likely fitting noise" or "OOS Sharpe survives and the
permutation test is significant, but small-N so treat cautiously". **A skeptical, honest
report is the point of this harness — do not oversell a good-looking equity curve.** Do not
tune the strategy's parameters against the out-of-sample split; treat the harness's own
walk-forward/permutation/CPCV results as the judge, not your own eyeballing of the curve.

## Do not

- Do not modify `tgtbt/engine.py`, `tgtbt/costs.py`, or anything under `tgtbt/validation/` —
  the strategy must work with the harness as-is.
- Do not commit or push. Leave the working tree for the human to review.
- Do not fetch data from anywhere except `tgtbt.data.get_prices` / `synthetic_prices`.
