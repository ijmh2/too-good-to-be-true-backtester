"""Individual chart builders. Each takes an optional `ax` so the scorecard can compose them.

Forms are chosen by the data's job: time series -> lines/area; a distribution vs an observed
value -> histogram + marker; a two-parameter score field -> a diverging heatmap centred at
zero; uncertainty over time -> a percentile fan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from tgtbt import TRADING_DAYS, metrics
from tgtbt.reporting import style as S


def _ax(ax):
    return ax if ax is not None else plt.subplots(figsize=(7, 4))[1]


def equity_curve(
    net_returns: pd.Series,
    benchmark: pd.Series | None = None,
    split_date=None,
    ax=None,
    title: str = "Equity curve (net of costs)",
):
    ax = _ax(ax)
    eq = (1 + net_returns.fillna(0)).cumprod()
    if benchmark is not None:
        bench = (1 + benchmark.reindex(net_returns.index).fillna(0)).cumprod()
        ax.plot(bench.index, bench.values, color=S.BENCHMARK, lw=1.6, label="Buy & hold")
    ax.plot(eq.index, eq.values, color=S.STRATEGY, lw=2.0, label="Strategy")

    if split_date is not None:
        sd = pd.Timestamp(split_date)
        ax.axvspan(eq.index[0], sd, color=S.BLUE, alpha=0.05)
        ax.axvline(sd, color=S.MUTED, lw=1.0, ls="--")
        # x in data coords, y in axes fraction -> labels pinned to the bottom, clear of the legend.
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(eq.index[0], 0.04, " in-sample", color=S.MUTED, fontsize=8, va="bottom",
                transform=trans)
        ax.text(sd, 0.04, " out-of-sample →", color=S.MUTED, fontsize=8, va="bottom",
                transform=trans)

    ax.set_yscale("log")
    ax.set_ylabel("growth of 1")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return ax


def drawdown(net_returns: pd.Series, ax=None):
    ax = _ax(ax)
    dd = metrics.drawdown_curve(net_returns) * 100
    ax.fill_between(dd.index, dd.values, 0, color=S.RED, alpha=0.25)
    ax.plot(dd.index, dd.values, color=S.RED, lw=1.2)
    ax.set_ylabel("drawdown %")
    ax.set_title(f"Underwater plot (max {dd.min():.1f}%)")
    return ax


def rolling_sharpe(net_returns: pd.Series, window: int = 126, ax=None):
    ax = _ax(ax)
    r = net_returns.fillna(0)
    rs = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(TRADING_DAYS)
    ax.axhline(0, color=S.MUTED, lw=0.9, ls="--")
    ax.plot(rs.index, rs.values, color=S.STRATEGY, lw=1.6)
    ax.set_ylabel("Sharpe")
    ax.set_title(f"Rolling {window}-day Sharpe")
    return ax


def parameter_heatmap(pivot: pd.DataFrame, ax=None, title: str = "Parameter surface (Sharpe)"):
    """Diverging heatmap centred at 0 — a broad plateau of colour = robust; a lone cell = overfit."""
    ax = _ax(ax)
    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) or 1.0
    im = ax.imshow(data, cmap=S.DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), labels=[str(c) for c in pivot.columns], fontsize=7)
    ax.set_yticks(range(pivot.shape[0]), labels=[str(i) for i in pivot.index], fontsize=7)
    ax.set_xlabel(pivot.columns.name)
    ax.set_ylabel(pivot.index.name)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color=S.INK if abs(v) < 0.6 * vmax else "#fcfcfb")
    ax.grid(False)
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return ax


def permutation_hist(perm, ax=None):
    ax = _ax(ax)
    ax.hist(perm.null_sharpes, bins=40, color=S.NULL, edgecolor=S.SURFACE, linewidth=0.4)
    ax.axvline(perm.real_sharpe, color=S.OBSERVED, lw=2.0)
    ax.text(perm.real_sharpe, ax.get_ylim()[1] * 0.92, f" observed\n p={perm.p_value:.3f}",
            color=S.OBSERVED, fontsize=8, va="top")
    ax.set_xlabel("Sharpe under random timing")
    ax.set_ylabel("count")
    ax.set_title(f"Permutation null ({perm.method})")
    return ax


def montecarlo_cone(bs, ax=None):
    ax = _ax(ax)
    b = bs.equity_bands
    x = np.arange(len(b))
    ax.fill_between(x, b["p5"], b["p95"], color=S.BLUE, alpha=0.12, label="5–95%")
    ax.fill_between(x, b["p25"], b["p75"], color=S.BLUE, alpha=0.22, label="25–75%")
    ax.plot(x, b["p50"], color=S.STRATEGY, lw=1.8, label="median")
    ax.set_yscale("log")
    ax.set_xlabel("trading days")
    ax.set_ylabel("growth of 1")
    ax.set_title("Monte-Carlo equity cone (block bootstrap)")
    ax.legend(loc="upper left")
    return ax


def metric_distribution(bs, metric: str = "sharpe", ax=None):
    ax = _ax(ax)
    s = bs.metric_samples[metric]
    lo, hi = bs.ci(metric, 0.95)
    ax.hist(s, bins=40, color=S.NULL, edgecolor=S.SURFACE, linewidth=0.4)
    ax.axvline(0, color=S.MUTED, lw=1.0, ls="--")
    ax.axvline(bs.point_estimates[metric], color=S.OBSERVED, lw=2.0)
    ax.axvspan(lo, hi, color=S.BLUE, alpha=0.08)
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ax.set_title(f"Bootstrap {metric} (95% CI [{lo:.2f}, {hi:.2f}])")
    return ax


def pbo_hist(pbo, ax=None):
    """Histogram of OOS-rank logits; mass left of 0 = in-sample winner failed out-of-sample."""
    ax = _ax(ax)
    lo = pbo.logits
    bins = np.linspace(min(lo.min(), -0.1), max(lo.max(), 0.1), 30)
    ax.hist(lo[lo < 0], bins=bins, color=S.RED, alpha=0.75, edgecolor=S.SURFACE, linewidth=0.4)
    ax.hist(lo[lo >= 0], bins=bins, color=S.BLUE, alpha=0.75, edgecolor=S.SURFACE, linewidth=0.4)
    ax.axvline(0, color=S.INK, lw=1.0)
    ax.set_xlabel("logit(OOS relative rank of IS-best)")
    ax.set_ylabel("count")
    ax.set_title(f"CSCV → PBO = {pbo.pbo:.2f}  ({pbo.verdict})")
    return ax


def degradation_scatter(pbo, ax=None):
    ax = _ax(ax)
    ax.scatter(pbo.is_perf, pbo.oos_perf, s=14, color=S.STRATEGY, alpha=0.5, edgecolor="none")
    lim = [min(pbo.is_perf.min(), pbo.oos_perf.min()), max(pbo.is_perf.max(), pbo.oos_perf.max())]
    ax.plot(lim, lim, color=S.MUTED, ls="--", lw=1.0)
    ax.axhline(0, color=S.GRID, lw=0.8)
    ax.set_xlabel("in-sample Sharpe of winner")
    ax.set_ylabel("out-of-sample Sharpe of winner")
    ax.set_title("Performance degradation (IS → OOS)")
    return ax
