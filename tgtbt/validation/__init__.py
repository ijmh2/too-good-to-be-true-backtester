"""Validation tools — the part that decides whether an edge is real or overfit.

Built up across phases:
- splits        : in-sample / out-of-sample date split
- gridutil      : parameter-grid expansion + a common evaluate() shim
- walkforward   : rolling re-selection of parameters, honest stitched OOS curve
- robustness    : parameter-sweep performance surface (plateau vs lucky spike)
- permutation   : random-timing / permuted-signal null distributions
- montecarlo    : block-bootstrap confidence intervals on the metrics
- deflated      : probabilistic & deflated Sharpe (multiple-testing correction)
- cpcv          : combinatorially-symmetric CV -> probability of backtest overfitting
"""

from tgtbt.validation.splits import train_test_split
from tgtbt.validation.gridutil import expand_grid, evaluate, config_returns
from tgtbt.validation.walkforward import walk_forward, WalkForwardResult
from tgtbt.validation.robustness import parameter_surface
from tgtbt.validation.permutation import permutation_test, PermutationResult
from tgtbt.validation.montecarlo import block_bootstrap, BootstrapResult
from tgtbt.validation.deflated import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)
from tgtbt.validation.cpcv import cscv_pbo, PBOResult

__all__ = [
    "train_test_split",
    "expand_grid",
    "evaluate",
    "config_returns",
    "walk_forward",
    "WalkForwardResult",
    "parameter_surface",
    "permutation_test",
    "PermutationResult",
    "block_bootstrap",
    "BootstrapResult",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "cscv_pbo",
    "PBOResult",
]
