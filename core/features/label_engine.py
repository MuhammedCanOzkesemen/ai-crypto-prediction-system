from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(
    close_series: pd.Series,
    horizon: int,
    threshold: float,
) -> pd.Series:
    if not isinstance(close_series, pd.Series):
        raise TypeError("close_series must be a pandas Series")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if threshold < 0.0:
        raise ValueError("threshold must be >= 0")
    if len(close_series) <= horizon:
        raise ValueError("close_series length must exceed horizon")

    safe_close = close_series.replace(0, np.nan)
    forward_return = close_series.shift(-horizon) / safe_close - 1.0

    labels = pd.Series(np.nan, index=close_series.index, dtype=float)
    labels[forward_return > threshold] = 1.0
    labels[forward_return < -threshold] = 0.0

    return labels.iloc[:-horizon]
