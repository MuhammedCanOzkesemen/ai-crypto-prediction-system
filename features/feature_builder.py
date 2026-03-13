"""
Feature builder for cryptocurrency price data.

Builds technical indicators and 14-day prediction targets from OHLCV DataFrames.
Uses only pandas and numpy (no external TA libraries).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.constants import (
    PRICE_COLUMNS,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BB_PERIOD,
    BB_NUM_STD,
    EMA_SHORT,
    EMA_LONG,
    SMA_SHORT,
    SMA_LONG,
    ATR_PERIOD,
    VOLATILITY_ROLLING,
    TARGET_HORIZON_DAYS,
    LAG_PERIODS,
)
from utils.logging_setup import get_logger

logger = get_logger(__name__)

REQUIRED_PRICE_COLUMNS = list(PRICE_COLUMNS)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _ensure_sorted_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure date column exists, is datetime, and DataFrame is sorted by date. Return copy."""
    out = df.copy()
    if "date" not in out.columns:
        raise ValueError("DataFrame must contain 'date' column")
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Technical indicators (pandas / numpy only)
# -----------------------------------------------------------------------------


def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD, column: str = "close") -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    close = df[column].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    return pd.Series(100 - (100 / (1 + rs)), index=df.index)


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
    column: str = "close",
) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    close = df[column].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": signal_line, "macd_histogram": macd_line - signal_line},
        index=df.index,
    )


def add_bollinger_bands(
    df: pd.DataFrame, period: int = BB_PERIOD, num_std: float = BB_NUM_STD, column: str = "close"
) -> pd.DataFrame:
    """Bollinger Bands (20, 2): middle = SMA(period), upper/lower = middle ± num_std * std."""
    close = df[column].astype(float)
    middle = close.rolling(window=period, min_periods=1).mean()
    std = close.rolling(window=period, min_periods=1).std()
    return pd.DataFrame(
        {"bb_upper": middle + num_std * std, "bb_middle": middle, "bb_lower": middle - num_std * std},
        index=df.index,
    )


def add_ema(df: pd.DataFrame, span: int, column: str = "close") -> pd.Series:
    """Exponential moving average."""
    return df[column].astype(float).ewm(span=span, adjust=False).mean()


def add_sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Simple moving average."""
    return df[column].astype(float).rolling(window=period, min_periods=1).mean()


def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Average True Range (Wilder). TR = max(high-low, |high-prev_close|, |low-prev_close|)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_rolling_volatility(df: pd.DataFrame, period: int = VOLATILITY_ROLLING, column: str = "close") -> pd.Series:
    """Rolling standard deviation of daily returns."""
    returns = df[column].astype(float).pct_change()
    return returns.rolling(window=period, min_periods=1).std()


def add_returns(df: pd.DataFrame, column: str = "close") -> pd.DataFrame:
    """Daily, 7-day, and 14-day returns."""
    close = df[column].astype(float)
    return pd.DataFrame(
        {
            "daily_return": close.pct_change(),
            "return_7d": close.pct_change(7),
            "return_14d": close.pct_change(14),
        },
        index=df.index,
    )


def add_volume_change(df: pd.DataFrame, column: str = "volume") -> pd.Series:
    """Day-over-day volume change. Inf/-inf replaced with NaN."""
    out = df[column].astype(float).pct_change()
    return out.replace([np.inf, -np.inf], np.nan)


def add_lag_features(
    df: pd.DataFrame, column: str, periods: tuple[int, ...] = LAG_PERIODS
) -> pd.DataFrame:
    """Add lagged series: column_lag_1, column_lag_2, ..."""
    s = df[column].astype(float)
    out = pd.DataFrame(index=df.index)
    for p in periods:
        out[f"{column}_lag_{p}"] = s.shift(p)
    return out


# -----------------------------------------------------------------------------
# Targets
# -----------------------------------------------------------------------------

TARGET_PRICE_COLUMNS = [f"target_price_{d}d" for d in range(1, TARGET_HORIZON_DAYS + 1)]


def add_targets(df: pd.DataFrame, horizon: int = TARGET_HORIZON_DAYS, column: str = "close") -> pd.DataFrame:
    """
    Add per-horizon target prices (target_price_1d .. target_price_14d) for multi-step forecasting.
    Also adds target_price_14d, target_return_14d, target_direction_14d for backward compatibility.
    """
    close = df[column].astype(float)
    out = {}
    for d in range(1, horizon + 1):
        out[f"target_price_{d}d"] = close.shift(-d)
    # Legacy 14d targets
    target_14 = close.shift(-horizon)
    target_return = (target_14 - close) / close
    target_direction = (target_return > 0).astype(np.int64)
    out["target_price_14d"] = target_14
    out["target_return_14d"] = target_return
    out["target_direction_14d"] = target_direction
    return pd.DataFrame(out, index=df.index)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def build_features(
    price_df: pd.DataFrame,
    *,
    include_targets: bool = True,
    target_horizon: int = TARGET_HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Build a clean, ML-ready feature DataFrame from a price DataFrame.

    Input: date, open, high, low, close, volume.

    Indicators: RSI(14), MACD(12,26,9), Bollinger Bands(20,2), EMA20, EMA50,
    SMA20, SMA50, ATR14, rolling_volatility_14, returns, volume_change,
    close_lag_1..7, daily_return_lag_1..7.

    Targets (if include_targets): target_price_14d, target_return_14d, target_direction_14d.

    Returns DataFrame with NaN rows dropped (ML-ready).
    """
    if price_df is None or price_df.empty:
        logger.warning("Empty price DataFrame provided")
        return pd.DataFrame()

    missing = set(REQUIRED_PRICE_COLUMNS) - set(price_df.columns)
    if missing:
        raise ValueError(f"Price DataFrame missing required columns: {missing}")

    df = _ensure_sorted_clean(price_df)

    # RSI(14)
    df["rsi_14"] = add_rsi(df, period=RSI_PERIOD)

    # MACD(12, 26, 9)
    macd = add_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    df["macd_line"] = macd["macd_line"]
    df["macd_signal"] = macd["macd_signal"]
    df["macd_histogram"] = macd["macd_histogram"]

    # Bollinger Bands(20, 2)
    bb = add_bollinger_bands(df, period=BB_PERIOD, num_std=BB_NUM_STD)
    df["bb_upper"] = bb["bb_upper"]
    df["bb_middle"] = bb["bb_middle"]
    df["bb_lower"] = bb["bb_lower"]

    # EMA 20, 50
    df["ema_20"] = add_ema(df, span=EMA_SHORT)
    df["ema_50"] = add_ema(df, span=EMA_LONG)

    # SMA 20, 50
    df["sma_20"] = add_sma(df, period=SMA_SHORT)
    df["sma_50"] = add_sma(df, period=SMA_LONG)

    # ATR(14)
    df["atr_14"] = add_atr(df, period=ATR_PERIOD)

    # Rolling volatility(14)
    df["rolling_volatility_14"] = add_rolling_volatility(df, period=VOLATILITY_ROLLING)

    # Returns
    ret = add_returns(df)
    df["daily_return"] = ret["daily_return"]
    df["return_7d"] = ret["return_7d"]
    df["return_14d"] = ret["return_14d"]

    # Volume change
    df["volume_change"] = add_volume_change(df)

    # Lag features (close and daily_return)
    for col, periods in [("close", LAG_PERIODS), ("daily_return", LAG_PERIODS)]:
        if col not in df.columns:
            continue
        lags = add_lag_features(df, col, periods)
        for c in lags.columns:
            df[c] = lags[c]

    # Targets
    if include_targets:
        tgt = add_targets(df, horizon=target_horizon)
        for c in tgt.columns:
            df[c] = tgt[c]

    out = df.dropna().reset_index(drop=True)
    logger.info("Built features: %d rows, %d columns (after dropna)", len(out), len(out.columns))
    return out


def get_feature_columns(include_targets: bool = False) -> list[str]:
    """Return list of feature column names (excluding date and raw OHLCV)."""
    base = [
        "rsi_14",
        "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower",
        "ema_20", "ema_50", "sma_20", "sma_50",
        "atr_14",
        "rolling_volatility_14",
        "daily_return", "return_7d", "return_14d",
        "volume_change",
    ]
    for col in ["close", "daily_return"]:
        for p in LAG_PERIODS:
            base.append(f"{col}_lag_{p}")
    if include_targets:
        base.extend(TARGET_PRICE_COLUMNS)
        base.extend(["target_return_14d", "target_direction_14d"])
    return base
