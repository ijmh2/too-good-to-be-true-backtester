"""The overfit scorecard — the repo's headline deliverable.

Runs the full validation gauntlet on a strategy and renders one figure that answers the only
question that matters: *is this edge real, or did the search just find a lucky fit?* The
verdict combines four independent signals, each of which can veto optimism:

- **Out-of-sample survival**  — walk-forward Sharpe (parameters re-picked as time moves on).
- **Timing significance**     — permutation p-value against a random-timing null.
- **Multiple-testing**        — the Deflated Sharpe (did the winner beat the best-of-N luck bar?).
- **Overfit probability**     — CSCV PBO (how often the in-sample winner fails out-of-sample).

No single green light is enough; the verdict is deliberately hard to please.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd

from tgtbt import metrics
from tgtbt.costs import CostModel
from tgtbt.data import to_returns
from tgtbt.reporting import plots
from tgtbt.reporting import style as S
from tgtbt.strategies.base import Strategy
from tgtbt.validation import (
    walk_forward,
    parameter_surface,
    permutation_test,
    block_bootstrap,
    config_returns,
    cscv_pbo,
)
from tgtbt.validation.deflated import deflated_sharpe_from_trials

# --- verdict thresholds (explicit and conservative) ---
PERM_ALPHA = 0.05      # permutation p below this = timing looks real
DSR_STRONG = 0.90      # deflated Sharpe above this = beats the multiple-testing bar
PBO_MAX = 0.50         # overfit probability above this = the search is fitting noise


@dataclass
class Scorecard:
    strategy_name: str
    full_summary: dict
    oos_summary: dict
    wf_result: object
    perm_result: object
    bootstrap: object
    deflated: dict
    pbo: object
    surface_pivot: pd.DataFrame
    net_returns: pd.Series
    benchmark: pd.Series | None
    split_date: object
    verdict: str = "inconclusive"
    flags: dict = field(default_factory=dict)

    # ---- verdict logic ----
    def _decide(self) -> None:
        oos_ok = self.wf_result.oos_sharpe > 0
        perm_ok = self.perm_result.p_value < PERM_ALPHA
        dsr_ok = (self.deflated.get("dsr") or 0) > DSR_STRONG
        pbo_ok = self.pbo.pbo < PBO_MAX
        self.flags = {
            "oos_positive": oos_ok,
            "timing_significant": perm_ok,
            "beats_deflated_bar": dsr_ok,
            "low_overfit_prob": pbo_ok,
        }
        passes = sum(self.flags.values())
        if not oos_ok or not pbo_ok:
            self.verdict = "likely overfit"
        elif passes == 4:
            self.verdict = "likely real edge"
        else:
            self.verdict = "inconclusive"

    # ---- text report ----
    def metric_lines(self) -> list[tuple[str, str]]:
        f, o = self.full_summary, self.oos_summary
        d = self.deflated
        rows = [
            ("Full-sample Sharpe", f"{f['sharpe']:.2f}"),
            ("Full-sample CAGR", f"{f['cagr']*100:.1f}%"),
            ("Max drawdown", f"{f['max_drawdown']*100:.1f}%"),
            ("Walk-forward OOS Sharpe", f"{self.wf_result.oos_sharpe:.2f}"),
            ("Fixed-config OOS Sharpe", f"{o['sharpe']:.2f}"),
            ("Permutation p-value", f"{self.perm_result.p_value:.3f}"),
            ("Bootstrap P(Sharpe>0)", f"{self.bootstrap.prob_positive('sharpe'):.2f}"),
            ("Trials searched (N)", f"{d['n_trials']}"),
            ("Deflation bar (SR*)", f"{d['sr_star']:.3f}"),
            ("Deflated Sharpe (DSR)", f"{d['dsr']:.2f}"),
            ("Prob. backtest overfit", f"{self.pbo.pbo:.2f}"),
        ]
        if self.benchmark is not None and "beta" in f:
            rows.insert(3, ("Beta vs benchmark", f"{f.get('beta', float('nan')):.2f}"))
        return rows

    def to_markdown(self) -> str:
        lines = [f"## Overfit scorecard — {self.strategy_name}", "",
                 f"**Verdict: {self.verdict.upper()}**", "",
                 "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in self.metric_lines()]
        lines += ["", "Verdict components:"]
        lines += [f"- {'✅' if v else '❌'} {k.replace('_', ' ')}" for k, v in self.flags.items()]
        return "\n".join(lines)

    # ---- the composed figure ----
    def figure(self):
        S.apply_style()
        fig = plt.figure(figsize=(15, 18))
        gs = fig.add_gridspec(5, 3, height_ratios=[0.45, 1.5, 1, 1, 1], hspace=0.42, wspace=0.26)

        # Row 0: verdict banner.
        banner = fig.add_subplot(gs[0, :])
        banner.axis("off")
        color = S.VERDICT_COLOR.get(self.verdict, S.MUTED)
        banner.text(0.0, 0.7, "too-good-to-be-true scorecard", fontsize=15, fontweight="bold",
                    color=S.INK, transform=banner.transAxes)
        banner.text(0.0, 0.18, self.strategy_name, fontsize=10, color=S.INK_2,
                    transform=banner.transAxes)
        banner.text(1.0, 0.55, self.verdict.upper(), fontsize=20, fontweight="bold", color=color,
                    ha="right", transform=banner.transAxes)
        chips = "   ".join(f"{'✓' if v else '✗'} {k.replace('_',' ')}" for k, v in self.flags.items())
        banner.text(1.0, 0.05, chips, fontsize=8, color=S.INK_2, ha="right",
                    transform=banner.transAxes)

        # Row 1: equity curve (span 2) + metrics table.
        ax_eq = fig.add_subplot(gs[1, :2])
        plots.equity_curve(self.net_returns, self.benchmark, self.split_date, ax=ax_eq)
        ax_tbl = fig.add_subplot(gs[1, 2])
        ax_tbl.axis("off")
        y = 0.98
        ax_tbl.text(0.0, y, "Metrics", fontsize=10, fontweight="bold", color=S.INK)
        y -= 0.09
        for k, v in self.metric_lines():
            ax_tbl.text(0.0, y, k, fontsize=8.5, color=S.INK_2)
            ax_tbl.text(1.0, y, v, fontsize=8.5, color=S.INK, ha="right",
                        fontfamily="monospace")
            y -= 0.083

        # Rows 2-4: the diagnostic panels.
        plots.drawdown(self.net_returns, ax=fig.add_subplot(gs[2, 0]))
        plots.rolling_sharpe(self.net_returns, ax=fig.add_subplot(gs[2, 1]))
        plots.parameter_heatmap(self.surface_pivot, ax=fig.add_subplot(gs[2, 2]))

        plots.permutation_hist(self.perm_result, ax=fig.add_subplot(gs[3, 0]))
        plots.montecarlo_cone(self.bootstrap, ax=fig.add_subplot(gs[3, 1]))
        plots.metric_distribution(self.bootstrap, "sharpe", ax=fig.add_subplot(gs[3, 2]))

        plots.pbo_hist(self.pbo, ax=fig.add_subplot(gs[4, 0]))
        plots.degradation_scatter(self.pbo, ax=fig.add_subplot(gs[4, 1]))
        ax_help = fig.add_subplot(gs[4, 2])
        ax_help.axis("off")
        ax_help.text(0.0, 0.95, "How to read the verdict", fontsize=9.5, fontweight="bold",
                     color=S.INK)
        note = (
            "A green verdict needs ALL four:\n"
            "• OOS Sharpe > 0 (walk-forward)\n"
            "• permutation p < 0.05\n"
            "• deflated Sharpe > 0.90\n"
            "• overfit prob (PBO) < 0.50\n\n"
            "Any failure of OOS or PBO forces\n"
            "'likely overfit'. The bar is meant\n"
            "to be hard to clear — that's the\n"
            "whole point."
        )
        ax_help.text(0.0, 0.82, note, fontsize=8.2, color=S.INK_2, va="top")
        return fig


def run_scorecard(
    strategy: Strategy,
    factory,
    grid: dict[str, list],
    prices: pd.DataFrame,
    benchmark: pd.Series | None = None,
    split_date=None,
    cost_model: CostModel | None = None,
    n_folds: int = 5,
    perm_n: int = 500,
    boot_n: int = 500,
    pbo_splits: int = 10,
) -> Scorecard:
    """Run every validation and assemble a Scorecard (call `.figure()` / `.to_markdown()`)."""
    cost_model = cost_model or CostModel()

    # Headline backtest + in/out-of-sample split for the fixed configuration.
    res = strategy.backtest(prices, cost_model=cost_model)
    net = res.net_returns
    if split_date is None:
        split_date = prices.index[int(len(prices) * 0.6)]
    oos_net = net.loc[pd.Timestamp(split_date) + pd.Timedelta(days=1):]
    full_summary = metrics.summary(net, benchmark)
    oos_summary = metrics.summary(oos_net, benchmark)

    # Walk-forward (honest OOS), permutation null, Monte-Carlo.
    wf = walk_forward(factory, grid, prices, n_folds=n_folds, cost_model=cost_model)
    perm = permutation_test(strategy, prices, n=perm_n, cost_model=cost_model)
    boot = block_bootstrap(net, n=boot_n, block=20)

    # Multiple-testing: the full trial matrix drives both deflated Sharpe and PBO.
    trial_matrix = config_returns(factory, grid, prices, cost_model)
    deflated = deflated_sharpe_from_trials(trial_matrix)
    pbo = cscv_pbo(trial_matrix, n_splits=pbo_splits)

    # Parameter surface: pivot the two most-varied parameters, others held mid-grid.
    varied = [k for k, v in grid.items() if len(v) > 1]
    two = (varied + [k for k in grid if k not in varied])[:2]
    reduced = {k: (grid[k] if k in two else [grid[k][len(grid[k]) // 2]]) for k in grid}
    surface = parameter_surface(factory, reduced, prices, cost_model=cost_model)
    pivot = surface.pivot(index=two[0], columns=two[1], values="score")

    card = Scorecard(
        strategy_name=strategy.name,
        full_summary=full_summary,
        oos_summary=oos_summary,
        wf_result=wf,
        perm_result=perm,
        bootstrap=boot,
        deflated=deflated,
        pbo=pbo,
        surface_pivot=pivot,
        net_returns=net,
        benchmark=benchmark,
        split_date=split_date,
    )
    card._decide()
    return card
