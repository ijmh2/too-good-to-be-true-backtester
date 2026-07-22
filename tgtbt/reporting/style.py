"""Central chart styling — one validated palette, applied everywhere.

Colours come from a validated categorical/diverging palette (worst adjacent-pair CVD ΔE well
clear of the 12 target). Assignment follows the *job* each colour does, not decoration:

- identity  -> strategy is blue; the benchmark is recessive muted grey
- polarity  -> diverging blue(+)/red(-) with a neutral grey midpoint at zero (Sharpe surfaces,
               PBO logits) so "good vs bad" is a hue opposition, never a rainbow
- magnitude -> a single blue ramp
- status    -> good / warning / critical, reserved for the verdict only

Everything renders on a light chart surface so the committed PNGs read correctly under both
GitHub light and dark themes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- surfaces & ink ---
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- categorical slots (fixed order = the CVD-safety mechanism) ---
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

# --- roles ---
STRATEGY = BLUE
BENCHMARK = MUTED
NULL = "#b9b8b2"        # null / random-timing distribution: recessive grey
OBSERVED = BLUE         # the real, observed statistic marked against a null

# --- status (verdict only) ---
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}
VERDICT_COLOR = {
    "likely real edge": STATUS["good"],
    "inconclusive": STATUS["warning"],
    "likely overfit": STATUS["critical"],
}

# Diverging blue<->red, neutral grey midpoint — for signed magnitudes centred at 0.
DIVERGING = LinearSegmentedColormap.from_list(
    "tgtbt_div", [RED, "#f0efec", BLUE]
)
# Single-hue blue ramp — for unsigned magnitude.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "tgtbt_seq", ["#cde2fb", BLUE, "#0d366b"]
)


def apply_style() -> None:
    """Set global matplotlib rcParams to the house style. Call once before plotting."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_2,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "figure.dpi": 130,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
        }
    )


def save_fig(fig, path: str) -> str:
    """Save with tight bounding box and the surface colour; return the path."""
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, dpi=130)
    return path
