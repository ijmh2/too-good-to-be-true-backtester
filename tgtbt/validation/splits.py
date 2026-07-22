"""In-sample / out-of-sample date splitting.

The cardinal rule: everything — parameter choices, thresholds, "having a look" — happens on
the in-sample slice; the out-of-sample slice is touched exactly once, at the end. These
helpers just make the split explicit and hard to fudge.
"""

from __future__ import annotations

import pandas as pd


def train_test_split(
    data: pd.DataFrame | pd.Series, split: str | pd.Timestamp
) -> tuple:
    """Split a time-indexed object at `split` (inclusive on the train side).

    Returns (in_sample, out_of_sample). `split` is the last date kept in-sample.
    """
    ts = pd.Timestamp(split)
    return data.loc[:ts], data.loc[ts + pd.Timedelta(days=1):]
