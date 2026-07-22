"""Performance metrics computed from a daily return series.

All annualised figures use 252 trading days. Sharpe/Sortino are computed on excess returns
over a (constant, annualised) risk-free rate, defaulting to 0 for simplicity — swap in a
real short-rate series later without touching call sites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tgtbt import TRADING_DAYS


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate implied by the return series."""
    r = returns.dropna()
    if len(r) == 0:
        return np.nan
    total_growth = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS
    if years <= 0 or total_growth <= 0:
        return np.nan
    return total_growth ** (1.0 / years) - 1.0


def ann_vol(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe ratio. `rf` is an annualised risk-free rate."""
    excess = returns.dropna() - rf / TRADING_DAYS
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sortino ratio (downside deviation in the denominator)."""
    excess = returns.dropna() - rf / TRADING_DAYS
    downside = excess[excess < 0]
    dd = np.sqrt((downside**2).mean()) if len(downside) else np.nan
    if not dd or np.isnan(dd):
        return np.nan
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def drawdown_curve(returns: pd.Series) -> pd.Series:
    """Running drawdown (<= 0) of the compounded equity curve."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_curve(returns)
    return float(dd.min()) if len(dd) else np.nan


def calmar(returns: pd.Series) -> float:
    """CAGR divided by the magnitude of max drawdown."""
    mdd = max_drawdown(returns)
    if not mdd or np.isnan(mdd):
        return np.nan
    return cagr(returns) / abs(mdd)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of active (non-zero) days that were positive."""
    active = returns[returns != 0].dropna()
    if len(active) == 0:
        return np.nan
    return float((active > 0).mean())


def alpha_beta(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """Annualised alpha and beta of `returns` regressed on `benchmark` (OLS).

    Returns (alpha_annualised, beta). Alpha is the daily intercept * 252.
    """
    df = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    if len(df) < 3:
        return (np.nan, np.nan)
    beta, intercept = np.polyfit(df["b"].to_numpy(), df["r"].to_numpy(), 1)
    return (float(intercept * TRADING_DAYS), float(beta))


def summary(returns: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Bundle the headline metrics into a dict (order = display order)."""
    out = {
        "cagr": cagr(returns),
        "ann_vol": ann_vol(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns),
        "hit_rate": hit_rate(returns),
        "n_days": int(returns.dropna().shape[0]),
    }
    if benchmark is not None:
        a, b = alpha_beta(returns, benchmark)
        out["alpha_ann"] = a
        out["beta"] = b
    return out
