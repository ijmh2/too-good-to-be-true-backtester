"""too-good-to-be-true-backtester — interactive UI.

Point it at a strategy (a built-in, or your own uploaded script) and it runs the full
overfit gauntlet and renders the verdict scorecard. Launch with:

    pip install -e ".[ui]"
    streamlit run app/app.py

An uploaded strategy is arbitrary Python executed locally in this process — only run scripts
you trust (this is a personal research tool, same trust model as a notebook).
"""

from __future__ import annotations

import importlib.util
import io
import tempfile

import pandas as pd
import streamlit as st

from tgtbt.costs import CostModel
from tgtbt.data import get_prices, synthetic_prices
from tgtbt.reporting import style as S
from tgtbt.reporting.scorecard import run_scorecard
from tgtbt.strategies import (
    BuyAndHold,
    DualMomentum,
    MeanReversion,
    TrendVolTarget,
    make_dual_momentum,
    make_mean_reversion,
    make_trend_vt,
)

# --- built-in strategy registry: (instance builder, factory, grid, default tickers) ---
BUILTINS = {
    "Trend + volatility target": dict(
        make=lambda: TrendVolTarget(trend_window=200, vol_window=20, target_vol=0.15),
        factory=make_trend_vt,
        grid={"trend_window": [50, 100, 150, 200], "vol_window": [20, 40, 60], "target_vol": [0.15]},
        tickers="SPY",
    ),
    "Short-horizon mean reversion": dict(
        make=lambda: MeanReversion(lookback=5, entry_z=-1.0),
        factory=make_mean_reversion,
        grid={"lookback": [3, 5, 10, 20], "entry_z": [-0.5, -1.0, -1.5, -2.0]},
        tickers="SPY",
    ),
    "Dual momentum (rotation)": dict(
        make=lambda: DualMomentum(lookback=126),
        factory=make_dual_momentum,
        grid={"lookback": [63, 126, 252]},
        tickers="SPY,TLT,EFA",
    ),
}

VERDICT_UI = {
    "likely real edge": ("✅", S.STATUS["good"]),
    "inconclusive": ("⚠️", S.STATUS["warning"]),
    "likely overfit": ("🚩", S.STATUS["critical"]),
}


@st.cache_data(show_spinner=False)
def load_prices(tickers: tuple[str, ...], start: str, end: str) -> tuple[pd.DataFrame, str]:
    try:
        px = get_prices(list(tickers), start=start, end=end)
        if px.dropna(how="all").shape[0] > 250:
            return px, "live (yfinance)"
        raise RuntimeError("insufficient rows")
    except Exception as exc:  # noqa: BLE001
        n = max(750, (pd.Timestamp(end) - pd.Timestamp(start)).days)
        px = pd.concat(
            [synthetic_prices(t, n_days=n, seed=i) for i, t in enumerate(tickers)], axis=1
        )
        return px, f"synthetic fallback ({exc})"


def load_uploaded_module(uploaded) -> object:
    """Execute an uploaded .py and hand back its module. Must define STRATEGY/FACTORY/GRID."""
    with tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False) as fh:
        fh.write(uploaded.getvalue())
        path = fh.name
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # noqa: S102 - trusted local script, by design
    for attr in ("STRATEGY", "FACTORY", "GRID"):
        if not hasattr(mod, attr):
            raise AttributeError(f"uploaded script must define a module-level `{attr}`")
    return mod


# ----------------------------------------------------------------------------------------
st.set_page_config(page_title="too-good-to-be-true-backtester", page_icon="🚩", layout="wide")

st.title("🚩 too-good-to-be-true-backtester")
st.caption(
    "Hand it a strategy; it runs out-of-sample splits, walk-forward, a parameter-robustness "
    "surface, permutation nulls, a Monte-Carlo bootstrap, the Deflated Sharpe and CSCV → "
    "probability of backtest overfitting, then returns one honest verdict."
)

with st.sidebar:
    st.header("Strategy")
    source = st.radio("Source", ["Built-in", "Upload a script"], horizontal=True)

    strategy = factory = grid = None
    tickers_default = "SPY"
    error = None

    if source == "Built-in":
        choice = st.selectbox("Choose a strategy", list(BUILTINS))
        spec = BUILTINS[choice]
        strategy, factory, grid = spec["make"](), spec["factory"], spec["grid"]
        tickers_default = spec["tickers"]
        st.caption(f"Parameter grid searched: `{grid}`")
    else:
        st.caption("Script must define `STRATEGY`, `FACTORY`, `GRID` (see the template in `app/`).")
        uploaded = st.file_uploader("Strategy .py", type="py")
        if uploaded is not None:
            try:
                mod = load_uploaded_module(uploaded)
                strategy, factory, grid = mod.STRATEGY, mod.FACTORY, mod.GRID
                tickers_default = getattr(mod, "TICKERS", "SPY")
                st.success(f"Loaded `{strategy.name}` — grid: `{grid}`")
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                st.error(f"Could not load script: {exc}")

    st.header("Universe & period")
    tickers_str = st.text_input("Tickers (comma-separated)", tickers_default)
    col_a, col_b = st.columns(2)
    start = col_a.text_input("Start", "2010-01-01")
    end = col_b.text_input("End", "2024-12-31")
    split = st.text_input("In/out-of-sample split date", "2021-12-31")

    st.header("Backtest settings")
    cost_bps = st.slider("Transaction cost (bps / one-way turnover)", 0.0, 25.0, 5.0, 0.5)
    fast = st.toggle("Fast mode (fewer resamples)", value=True)
    perm_n = 300 if fast else 1000
    boot_n = 300 if fast else 1000
    n_folds = st.slider("Walk-forward folds", 3, 8, 5)

    run = st.button("Run the gauntlet", type="primary", use_container_width=True,
                    disabled=strategy is None)

if strategy is None:
    st.info("Pick a built-in strategy or upload a script in the sidebar, then **Run the gauntlet**.")
    st.stop()

if run:
    tickers = tuple(t.strip().upper() for t in tickers_str.split(",") if t.strip())
    with st.spinner("Loading prices…"):
        prices, src = load_prices(tickers, start, end)
        benchmark = BuyAndHold().backtest(prices).net_returns
    st.caption(f"Data: {src} · {len(prices)} rows · {prices.index[0].date()} → {prices.index[-1].date()}")

    with st.spinner("Running walk-forward, permutation, Monte-Carlo, CPCV…"):
        card = run_scorecard(
            strategy, factory, grid, prices, benchmark=benchmark, split_date=split,
            cost_model=CostModel(cost_bps), perm_n=perm_n, boot_n=boot_n, n_folds=n_folds,
        )

    icon, color = VERDICT_UI.get(card.verdict, ("•", S.MUTED))
    st.markdown(
        f"<h2 style='color:{color}'>{icon} Verdict: {card.verdict.upper()}</h2>",
        unsafe_allow_html=True,
    )

    # Headline metrics.
    f = card.full_summary
    cols = st.columns(5)
    cols[0].metric("Full Sharpe", f"{f['sharpe']:.2f}")
    cols[1].metric("Walk-forward OOS Sharpe", f"{card.wf_result.oos_sharpe:.2f}")
    cols[2].metric("Permutation p", f"{card.perm_result.p_value:.3f}")
    cols[3].metric("Deflated Sharpe", f"{card.deflated['dsr']:.2f}")
    cols[4].metric("Overfit prob (PBO)", f"{card.pbo.pbo:.2f}")

    # Verdict components.
    chips = "  ".join(
        f"{'✅' if v else '❌'} {k.replace('_', ' ')}" for k, v in card.flags.items()
    )
    st.write(chips)

    # The composed scorecard figure.
    with st.spinner("Rendering scorecard…"):
        fig = card.figure()
    st.pyplot(fig, use_container_width=True)

    # Downloads + full report.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=S.SURFACE)
    d1, d2 = st.columns(2)
    d1.download_button("Download scorecard PNG", buf.getvalue(),
                       file_name=f"scorecard_{strategy.name}.png", mime="image/png")
    d2.download_button("Download markdown report", card.to_markdown(),
                       file_name=f"scorecard_{strategy.name}.md", mime="text/markdown")
    with st.expander("Full markdown report"):
        st.markdown(card.to_markdown())
