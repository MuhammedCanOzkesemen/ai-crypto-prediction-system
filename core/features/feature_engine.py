import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # -------------------------------------------------------------------------
    # Momentum: log returns, backward-looking only
    # -------------------------------------------------------------------------
    log_ret = np.log(close / close.shift(1))
    return_1d = log_ret
    return_5d = np.log(close / close.shift(5))
    return_10d = np.log(close / close.shift(10))
    return_20d = np.log(close / close.shift(20))

    # -------------------------------------------------------------------------
    # Volatility: rolling std of log returns
    # -------------------------------------------------------------------------
    realized_vol_5d = log_ret.rolling(5, min_periods=5).std()
    realized_vol_20d = log_ret.rolling(20, min_periods=20).std()
    vol_ratio = (realized_vol_5d / realized_vol_20d).replace(
        [np.inf, -np.inf], np.nan
    )

    # -------------------------------------------------------------------------
    # RSI(14) via Wilder smoothing, scaled to [0, 1]
    # -------------------------------------------------------------------------
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(com=13, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14 = (1 - 1 / (1 + rs)).clip(0.0, 1.0)

    # -------------------------------------------------------------------------
    # Bollinger Band position: (close - lower) / (upper - lower), clipped [0,1]
    # -------------------------------------------------------------------------
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_width = (bb_upper - bb_lower).replace(0, np.nan)
    bb_position = ((close - bb_lower) / bb_width).clip(0.0, 1.0)

    # -------------------------------------------------------------------------
    # Close vs SMA20: percentage deviation from 20-day mean
    # -------------------------------------------------------------------------
    close_vs_sma20 = (close / sma20.replace(0, np.nan) - 1).replace(
        [np.inf, -np.inf], np.nan
    )

    # -------------------------------------------------------------------------
    # Volume: normalized ratios, both backward-looking
    # -------------------------------------------------------------------------
    vol_sma20 = volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    vol_sma5 = volume.rolling(5, min_periods=5).mean().replace(0, np.nan)
    volume_ratio = (volume / vol_sma20).replace([np.inf, -np.inf], np.nan)
    volume_trend = (vol_sma5 / vol_sma20).replace([np.inf, -np.inf], np.nan)

    # -------------------------------------------------------------------------
    # Intraday structure: both use same-bar OHLC, known at market close
    # -------------------------------------------------------------------------
    body_pct = ((close - open_) / open_.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    hl_range_pct = ((high - low) / open_.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # -------------------------------------------------------------------------
    # Time: cyclic day-of-week encoding
    # -------------------------------------------------------------------------
    if isinstance(df.index, pd.DatetimeIndex):
        weekday = df.index.dayofweek.astype(float)
        weekday = pd.Series(weekday, index=df.index)
    elif "date" in df.columns:
        weekday = pd.to_datetime(df["date"]).dt.dayofweek.astype(float)
        weekday = weekday.values
        weekday = pd.Series(weekday, index=df.index)
    else:
        raise ValueError(
            "df must have a DatetimeIndex or a 'date' column to compute day features"
        )

    day_sin = np.sin(2 * np.pi * weekday / 7)
    day_cos = np.cos(2 * np.pi * weekday / 7)

    # -------------------------------------------------------------------------
    # Assemble output: exactly 16 features, no raw price columns
    # -------------------------------------------------------------------------
    out = pd.DataFrame(
        {
            "return_1d": return_1d,
            "return_5d": return_5d,
            "return_10d": return_10d,
            "return_20d": return_20d,
            "realized_vol_5d": realized_vol_5d,
            "realized_vol_20d": realized_vol_20d,
            "vol_ratio": vol_ratio,
            "rsi_14": rsi_14,
            "bb_position": bb_position,
            "close_vs_sma20": close_vs_sma20,
            "volume_ratio": volume_ratio,
            "volume_trend": volume_trend,
            "body_pct": body_pct,
            "hl_range_pct": hl_range_pct,
            "day_sin": day_sin,
            "day_cos": day_cos,
        },
        index=df.index,
    )

    return out
