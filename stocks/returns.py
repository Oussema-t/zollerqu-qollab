"""Weekly log returns and the chronological train/test split (NumPy only)."""
import math
from dataclasses import dataclass

import numpy as np

DEFAULT_TRAIN_FRACTION = 0.70
# 2008 crash window used throughout: week before Lehman failed -> week before the stock-market low.
CRASH_START = np.datetime64("2008-09-12")
CRASH_END = np.datetime64("2009-03-06")


def log_returns(prices):
    """r[t] = ln(P[t] / P[t-1]) along axis 0. Rejects non-finite or non-positive prices."""
    p = np.asarray(prices, dtype=float)
    if p.ndim not in (1, 2) or p.shape[0] < 2:
        raise ValueError(f"need at least 2 prices along axis 0, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("prices must be finite and positive to take logarithms")
    return np.diff(np.log(p), axis=0)


@dataclass(frozen=True)
class Split:
    n_rows: int
    train_fraction: float
    split_index: int  # first test row; training = rows [0, split_index)

    @property
    def train(self):
        return slice(0, self.split_index)

    @property
    def test(self):
        return slice(self.split_index, self.n_rows)


def chronological_split(n_rows, train_fraction=DEFAULT_TRAIN_FRACTION):
    """First floor(train_fraction * n_rows) rows train, the rest test. Never shuffled."""
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")
    # round() first so that e.g. 0.7 * 10 = 7.000000000000001 still gives 7
    split_index = math.floor(round(train_fraction * n_rows, 9))
    if not 1 <= split_index < n_rows:
        raise ValueError(f"{n_rows} rows with train_fraction {train_fraction} leave an empty block")
    return Split(int(n_rows), float(train_fraction), split_index)


def window_rows(return_dates, start=CRASH_START, end=CRASH_END):
    """Indices of the returns that make up the move from the price at `start` to the price at `end`.

    Return row t is labelled with its ending date, so these are the rows ending in (start, end].
    """
    d = np.asarray(return_dates, dtype="datetime64[D]")
    return np.flatnonzero((d > start) & (d <= end))
