"""Turn a Scorecard into a JSON-safe payload for the web frontend.

We return three things: the verdict + component flags, a flat metrics list (ready to render as
tiles/table), and enough chart data for the frontend to draw an interactive equity curve — plus
the full composed scorecard as a base64 PNG (the portfolio-grade figure, reused as-is).
"""

from __future__ import annotations

import base64
import io
import math

import pandas as pd

from tgtbt.reporting import style as S


def _num(x, nd: int = 2):
    """Round to a JSON-safe number; NaN/inf -> None (bare NaN is invalid JSON)."""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def _downsample(s: pd.Series, max_points: int = 800) -> pd.Series:
    step = max(1, len(s) // max_points)
    return s.iloc[::step]


def scorecard_to_dict(card, include_png: bool = True) -> dict:
    net = card.net_returns.fillna(0.0)
    eq = (1 + net).cumprod()
    bench = card.benchmark
    bench_eq = (1 + bench.reindex(net.index).fillna(0.0)).cumprod() if bench is not None else None

    eq_ds = _downsample(eq)
    equity = {
        "dates": [d.strftime("%Y-%m-%d") for d in eq_ds.index],
        "strategy": [round(float(v), 5) for v in eq_ds.values],
    }
    if bench_eq is not None:
        equity["benchmark"] = [round(float(v), 5) for v in _downsample(bench_eq).values]

    payload = {
        "strategy_name": card.strategy_name,
        "verdict": card.verdict,
        "flags": card.flags,
        "metrics": [{"label": k, "value": v} for k, v in card.metric_lines()],
        "tiles": {
            "full_sharpe": _num(card.full_summary["sharpe"]),
            "wf_oos_sharpe": _num(card.wf_result.oos_sharpe),
            "perm_p": _num(card.perm_result.p_value, 3),
            "dsr": _num(card.deflated["dsr"]),
            "pbo": _num(card.pbo.pbo),
            "max_drawdown_pct": _num(float(card.full_summary["max_drawdown"]) * 100, 1),
        },
        "split_date": str(pd.Timestamp(card.split_date).date()),
        "equity": equity,
    }

    if include_png:
        fig = card.figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=S.SURFACE)
        import matplotlib.pyplot as plt

        plt.close(fig)  # free the figure; API may be long-lived
        payload["scorecard_png"] = "data:image/png;base64," + base64.b64encode(
            buf.getvalue()
        ).decode()
    return payload
