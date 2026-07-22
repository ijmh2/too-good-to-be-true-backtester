"""Combinatorially-Symmetric Cross-Validation -> Probability of Backtest Overfitting (PBO).

(Bailey, Borwein, Lopez de Prado, Zhu, 2017.) The headline overfit diagnostic, and the one
the repo is named after.

Given a matrix of net returns — one column per configuration you tried — cut the timeline
into S blocks and, over *every* way of splitting the blocks into an in-sample half and an
out-of-sample half, ask one question: does the configuration that looked best in-sample also
beat the median out-of-sample? When the in-sample winner is repeatedly a below-median
performer out-of-sample, your selection process is fitting noise.

PBO = the fraction of splits where the in-sample-best configuration lands in the bottom half
out-of-sample. A high PBO (say > 0.5) means the whole search is more likely to have found a
lucky fit than a real edge — regardless of how good the best backtest looked.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tgtbt import metrics


@dataclass
class PBOResult:
    pbo: float                     # probability of backtest overfitting in [0, 1]
    logits: np.ndarray             # logit of the OOS relative rank of the IS-best, per split
    is_perf: np.ndarray            # IS Sharpe of the selected config, per split
    oos_perf: np.ndarray           # OOS Sharpe of the selected config, per split
    n_splits: int
    n_combinations: int

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.pbo):
            return "inconclusive"
        return "likely overfit" if self.pbo > 0.5 else "holds up"


def _perf(matrix: np.ndarray) -> np.ndarray:
    """Per-column Sharpe of a (rows x configs) return block (annualisation cancels in ranks)."""
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mean / sd, -np.inf)


def cscv_pbo(
    trial_returns: pd.DataFrame,
    n_splits: int = 10,
    max_combinations: int = 2000,
    seed: int = 0,
) -> PBOResult:
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    M = trial_returns.dropna().to_numpy()
    T, N = M.shape
    if N < 2:
        raise ValueError("need at least two configurations to assess overfitting")

    # Cut rows into S contiguous, near-equal blocks.
    block_ids = np.array_split(np.arange(T), n_splits)

    all_combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    rng = np.random.default_rng(seed)
    if len(all_combos) > max_combinations:
        pick = rng.choice(len(all_combos), size=max_combinations, replace=False)
        all_combos = [all_combos[i] for i in pick]

    logits, is_perf, oos_perf = [], [], []
    for is_blocks in all_combos:
        is_set = set(is_blocks)
        is_rows = np.concatenate([block_ids[b] for b in range(n_splits) if b in is_set])
        oos_rows = np.concatenate([block_ids[b] for b in range(n_splits) if b not in is_set])

        is_sharpe = _perf(M[is_rows])
        oos_sharpe = _perf(M[oos_rows])

        best = int(np.argmax(is_sharpe))               # in-sample winner
        sel_oos = oos_sharpe[best]

        # Relative rank of the winner within the OOS Sharpes, in (0, 1); 1 = best OOS.
        less = np.sum(oos_sharpe < sel_oos)
        equal = np.sum(oos_sharpe == sel_oos)
        omega = (less + 0.5 * equal) / N
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(np.log(omega / (1 - omega)))
        is_perf.append(is_sharpe[best])
        oos_perf.append(sel_oos)

    logits = np.asarray(logits)
    pbo = float((logits < 0).mean())
    return PBOResult(
        pbo=pbo,
        logits=logits,
        is_perf=np.asarray(is_perf),
        oos_perf=np.asarray(oos_perf),
        n_splits=n_splits,
        n_combinations=len(all_combos),
    )
