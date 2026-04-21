import pandas as pd


def build_cross_sectional_labels(
    all_close: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    forward_returns = all_close.pct_change(horizon).shift(-horizon)
    median_return = forward_returns.median(axis=1)
    labels = forward_returns.gt(median_return, axis=0).astype(int)
    return labels.dropna()
