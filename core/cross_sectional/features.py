import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    f = pd.DataFrame(index=df.index)

    returns = close.pct_change()
    f["returns"] = returns
    f["log_volume"] = np.log1p(volume)
    f["hl_range"] = (high - low) / close
    f["close_open"] = (close - df["open"]) / df["open"]

    for lag in [1, 2, 3, 5]:
        f[f"ret_lag_{lag}"] = returns.shift(lag)

    vol_5d = returns.rolling(5).std()
    vol_20d = returns.rolling(20).std()
    f["realized_vol_5d"] = vol_5d
    f["return_5d"] = close.pct_change(5)

    f["momentum_accel"] = close.pct_change(3) - close.pct_change(7)
    f["vol_shock"] = vol_5d / vol_20d.replace(0, np.nan)

    hl_5d = high.rolling(5).max() - low.rolling(5).min()
    hl_20d = high.rolling(20).max() - low.rolling(20).min()
    f["range_compression"] = hl_5d / hl_20d.replace(0, np.nan)

    sma20 = close.rolling(20).mean()
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    f["trend_strength"] = (close - sma20).abs() / atr.replace(0, np.nan)

    f["breakout"] = (close > high.shift(1).rolling(10).max()).astype(float)

    sma10 = close.rolling(10).mean()
    std10 = close.rolling(10).std()
    f["mean_reversion"] = (close - sma10) / std10.replace(0, np.nan)

    vol_ma30 = volume.rolling(30).mean()
    f["volume_spike"] = volume / vol_ma30.replace(0, np.nan)

    return f.dropna()
