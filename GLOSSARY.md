# Glossary

Every term shown in the CLI output, the web UI, and the scorecard PNG, defined precisely to
match what this repo actually computes — not a generic textbook version. If a number in your
own run doesn't match the intuition below, the code is the final authority; file references are
given throughout.

## The verdict

**Likely real edge / Inconclusive / Likely overfit** — the one-line answer
(`tgtbt/reporting/scorecard.py::_decide`). Built from four checks, split into two kinds:

- **Hard vetoes** (fail either → automatic `likely overfit`, no matter what else looks good):
  - **`oos_positive`** — did the walk-forward-selected strategy make money out-of-sample at
    all? If not, there's nothing left to be confident *about*.
  - **`low_overfit_prob`** — is the *search process* trustworthy (see PBO below)? A search
    that's more likely to have picked noise than signal is a disqualifier regardless of how
    the winning config looks.
- **Soft gates** (needed for the strongest verdict, but their absence alone only caps you at
  `inconclusive` — it never demotes you to `overfit`):
  - **`timing_significant`** — is the permutation p-value below 0.05?
  - **`beats_deflated_bar`** — is the Deflated Sharpe above 0.90?

Rationale: the hard vetoes ask "did this actually work, honestly" (validity); the soft gates
ask "how statistically sure are we" (confidence). A strategy can honestly survive
out-of-sample with a trustworthy search yet still lack the statistical power to hit the
stricter significance bars — a common, honest outcome with a modest number of events. That's
exactly what `inconclusive` is for; it is not a euphemism for `overfit`.

**Important:** the verdict certifies the *timing signal* is statistically genuine. It does
**not** certify the strategy beats a fully-invested benchmark — see Alpha/Beta below.

## Performance metrics (`tgtbt/metrics.py`)

- **Sharpe (ratio)** — mean daily excess return ÷ daily volatility, annualised (×√252).
  The standard risk-adjusted-return number; roughly, "return per unit of bumpiness." Above
  ~1.0 is strong for a daily strategy; 0.3–0.7 is common and not embarrassing.
- **Sortino (ratio)** — like Sharpe, but the denominator only penalises *downside* volatility
  (squared negative returns, divided by the total number of periods — up days count as zero
  penalty, not as removed from the average). Rewards strategies whose volatility is mostly on
  the upside.
- **CAGR** — compound annual growth rate: the constant annual return that would turn 1.0 into
  the strategy's actual final value over the actual number of years tested. The "raw return"
  headline number — deliberately *not* what the verdict is judged on (see the Alpha note).
- **Max drawdown** — the largest peak-to-trough decline in the compounded equity curve, e.g.
  −25% means the strategy was once worth 25% less than its prior best. The main "how bad could
  this feel to hold" number.
- **Calmar (ratio)** — CAGR ÷ |max drawdown|. Return per unit of worst-case pain.
- **Hit rate** — fraction of *active* days (non-zero position) that were profitable. Easy to
  over-interpret: a strategy with a low hit rate and rare large wins can still have a great
  Sharpe (trend-following often looks like this).
- **Annualised volatility** — standard deviation of daily returns, annualised. The
  denominator inside Sharpe, reported on its own too.

## Benchmark-relative (`tgtbt/metrics.py::alpha_beta`)

Computed only when a benchmark (buy-and-hold on the same asset) is supplied — an OLS
regression of the strategy's daily returns on the benchmark's.

- **Beta** — how much the strategy moves with the benchmark. Beta 0.4 means "on average,
  about 40% as much market exposure as being fully invested." A trend/vol-target strategy
  that's frequently in cash or scaled down will show a beta well under 1.0 — that's by
  design, not a flaw.
- **Alpha (annualised)** — the return left over *after* accounting for that beta exposure —
  the daily regression intercept, annualised (×252). **This is the number that resolves "how
  can this be a real edge if it made less money than buy-and-hold?"** A lower-beta strategy
  will usually have lower raw CAGR than a fully-invested benchmark in a rising market even
  when its alpha is genuinely positive — beta and alpha answer different questions than CAGR
  does. Positive alpha means the strategy is doing something with genuine skill, exposure-
  adjusted; it says nothing about which one made more absolute money.

## Out-of-sample & search integrity

- **In/out-of-sample split** — the date that divides "the search was allowed to look at this"
  (in-sample) from "the search was never allowed to look at this" (out-of-sample). Defaults
  to ~60% of the way through the data if not set explicitly.
- **Walk-forward OOS Sharpe** (`tgtbt/validation/walkforward.py`) — the honest one. The
  timeline is cut into folds; for each fold, the best parameter configuration is chosen using
  only data up to that point (never peeking forward), then applied to the *next*, still-unseen
  fold. Stitching together only those unseen-fold returns gives a Sharpe with zero look-ahead
  in the parameter choice itself — the closest thing here to "would this have worked live."
- **Fixed-config OOS Sharpe** — much simpler and weaker: just the headline (fixed) parameter
  choice's Sharpe, computed only on the out-of-sample slice. No re-selection, so it's an easier
  bar to pass than walk-forward — shown for comparison, not as the main OOS check.
- **Trials searched (N)** — the number of parameter combinations swept (the grid size). This
  number is what the Deflated Sharpe correction below is charging you for.

## Statistical significance

- **Permutation p-value** (`tgtbt/validation/permutation.py`) — keeps the strategy's actual
  positions (so its exposure and turnover are identical) but randomly rolls *when* those
  positions line up with the market's returns, many times, to build a null distribution of
  "Sharpe achievable by random timing alone, at the same exposure." The p-value is the
  fraction of those random-timing trials that matched or beat the real result. Small (< 0.05)
  means the specific timing is doing something a coin-flip-timed version of the same strategy
  couldn't.
- **Bootstrap P(Sharpe>0)** (`tgtbt/validation/montecarlo.py`) — resamples the return series
  in blocks (preserving short-run autocorrelation) to build a distribution of plausible
  Sharpes, then reports what fraction of that distribution is positive. Different question
  from the permutation test: this asks "how much is the *point estimate itself* just noise,"
  not "is the timing distinguishable from randomness."

## Overfitting diagnostics — the repo's namesake checks

- **Deflation bar (SR\*)** and **Deflated Sharpe Ratio (DSR)**
  (`tgtbt/validation/deflated.py`, Bailey & López de Prado) — a Sharpe ratio is a noisy
  estimate, and it's biased upward by *how many configurations you tried* before reporting the
  best one. SR\* is the Sharpe you'd expect to see **by pure luck** as the best of N trials
  (Trials searched, above) with the observed spread of results. DSR is then the probability
  the true Sharpe exceeds that inflated, luck-adjusted bar — not just zero. A DSR of 0.99 with
  only a handful of trials searched is a much weaker claim than the same DSR with fifty.
- **Probability of Backtest Overfitting (PBO)**
  (`tgtbt/validation/cpcv.py`, CSCV — Combinatorially-Symmetric Cross-Validation) — the
  headline diagnostic. Splits the timeline into blocks, and over *every* way of dividing those
  blocks into an in-sample half and an out-of-sample half, checks: does the configuration that
  looked best in-sample also perform above the out-of-sample median? PBO is the fraction of
  splits where it does *not* — i.e. the in-sample winner turns out to be a below-median
  performer out-of-sample. PBO ≈ 0.5 is what pure noise looks like (the in-sample winner is a
  coin flip out-of-sample); PBO > 0.5 means the search is *more likely than not* to have
  picked noise over signal, regardless of how good the winning backtest looked. This is the
  check that can flag a strategy "likely overfit" even when its raw Sharpe looks fine.

## Practical settings

- **Cost (bps)** — round-trip transaction cost assumption, charged per unit of *turnover*
  (`tgtbt/costs.py`): one full buy-then-sell round trip costs `2 × cost_bps`. 5 bps is a
  realistic-to-conservative default for liquid US equities/ETFs.
- **Walk-forward folds (`n_folds`)** — how many contiguous re-selection windows the walk-forward
  check uses. More folds = a stricter, more granular OOS test, but each fold has less data to
  select parameters on.
- **Fast vs. thorough** (`--thorough` / the UI toggle) — 300 vs. 1000 resamples for the
  permutation and bootstrap tests. More resamples give a smoother, slightly more precise null
  distribution; 300 is already enough to distinguish p ≈ 0.05 from p ≈ 0.5 reliably, so `fast`
  is the sensible default, not a corner cut.
- **Data source label** — `live (yfinance)` (real Yahoo Finance data), `synthetic fallback`
  (Yahoo was unreachable/rate-limited, so a random-walk stand-in was used — see the caveats in
  the README), or `uploaded CSV` (your own data via `--price-csv` or the browser upload).

## Scorecard figure panels

- **Equity curve** — growth of 1 unit invested, strategy vs. buy-and-hold, log scale, with the
  in-sample region shaded and the out-of-sample split marked.
- **Underwater plot** — the drawdown curve over time (0 = at a new high, negative = below the
  prior peak).
- **Rolling Sharpe** — Sharpe computed on a moving window (126 trading days ≈ 6 months), to
  show whether the edge is stable over time or concentrated in one period.
- **Parameter surface** — a heatmap of Sharpe across two of the swept parameters. A broad
  plateau of similar colour = the result is robust to small parameter changes (good sign); an
  isolated bright spot surrounded by very different neighbours = likely a lucky, overfit pick.
- **Permutation null (histogram)** — the distribution of Sharpes achieved by randomly-timed
  versions of the same strategy, with the real result marked — visually, is the blue line out
  in the tail, or buried in the middle of the grey pile?
- **Monte-Carlo equity cone** — a fan of plausible alternative equity curves from the block
  bootstrap, showing how much of the "growth" could plausibly have looked very different by
  chance alone.
- **CSCV → PBO (histogram)** — for every in-sample/out-of-sample split, whether the in-sample
  winner ranked above (blue) or below (red) the out-of-sample median. Mostly red = high PBO =
  the search is unreliable.
- **Performance degradation (IS → OOS)** — a scatter of in-sample vs. out-of-sample Sharpe for
  every split's winning configuration. Points clustered near the diagonal = performance
  survives the transition; points below the diagonal = performance decays out-of-sample
  (the norm for an overfit search).
