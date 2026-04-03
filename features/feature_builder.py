"""
Feature builder for cryptocurrency price data.

Builds technical indicators and 14-day prediction targets from OHLCV DataFrames.
Uses only pandas and numpy (no external TA libraries).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.constants import (
    ADX_PERIOD,
    BB_NUM_STD,
    BB_PERIOD,
    ATR_PERIOD,
    EMA_LONG,
    EMA_SHORT,
    LAG_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    PRICE_COLUMNS,
    ROC_PERIODS,
    RSI_PERIOD,
    SMA_LONG,
    SMA_SHORT,
    TARGET_HORIZON_DAYS,
    VOL_REGIME_LOGBOUND_HIGH,
    VOL_REGIME_LOGBOUND_LOW,
    VOLATILITY_ROLLING,
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
    """Daily, 3/7/14-day simple returns."""
    close = df[column].astype(float)
    return pd.DataFrame(
        {
            "daily_return": close.pct_change(),
            "return_3d": close.pct_change(3),
            "return_7d": close.pct_change(7),
            "return_14d": close.pct_change(14),
        },
        index=df.index,
    )


def add_rate_of_change(df: pd.DataFrame, periods: tuple[int, ...] = ROC_PERIODS, column: str = "close") -> pd.DataFrame:
    """ROC: (close / close.shift(n)) - 1."""
    close = df[column].astype(float)
    out = pd.DataFrame(index=df.index)
    for n in periods:
        out[f"roc_{n}"] = close.pct_change(n)
    return out


def add_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """ADX, +DI, -DI (Wilder-style smoothing via EWM alpha=1/period)."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=df.index, dtype=float)
    alpha = 1.0 / float(period)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    sm_plus = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    sm_minus = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * (sm_plus / (atr + 1e-12))
    minus_di = 100.0 * (sm_minus / (atr + 1e-12))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return pd.DataFrame(
        {
            f"adx_{period}": adx,
            f"plus_di_{period}": plus_di,
            f"minus_di_{period}": minus_di,
        },
        index=df.index,
    )


def _vol_regime_from_ewma(ewma_vol: pd.Series) -> pd.Series:
    """0=low, 1=medium, 2=high log-return volatility."""
    v = ewma_vol.astype(float)
    return pd.Series(
        np.select(
            [v < VOL_REGIME_LOGBOUND_LOW, v < VOL_REGIME_LOGBOUND_HIGH],
            [0, 1],
            default=2,
        ),
        index=v.index,
        dtype=float,
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
# Targets (single-step log return — multi-day path via recursive forecast at inference)
# -----------------------------------------------------------------------------

TARGET_LOG_RETURN_1D = "target_log_return_1d"
# Kept empty for imports that still reference the name; training uses TARGET_LOG_RETURN_1D only.
TARGET_PRICE_COLUMNS: list[str] = []


def add_targets(df: pd.DataFrame, horizon: int = TARGET_HORIZON_DAYS, column: str = "close") -> pd.DataFrame:
    """
    Add target_log_return_1d = log(close_{t+1} / close_t) for one-step-ahead training.
    """
    close = df[column].astype(float)
    ratio = close.shift(-1) / close.replace(0, np.nan)
    log_ret_fwd = np.log(np.clip(ratio, 1e-12, None))
    out = {TARGET_LOG_RETURN_1D: pd.Series(log_ret_fwd, index=df.index)}
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

    Indicators: RSI, MACD, BB, EMA/SMA, ATR, rolling vol, 3/7/14d returns, ROC, ADX/DI,
    log-return momentum sums, rolling/EWMA log vol (14/30), vol regime, EMA cross, interactions,
    log_close + lags, daily_return lags, volume_change.

    Targets (if include_targets): target_log_return_1d (next-day log return).

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

    # Returns (incl. 3d)
    ret = add_returns(df)
    df["daily_return"] = ret["daily_return"]
    df["return_3d"] = ret["return_3d"]
    df["return_7d"] = ret["return_7d"]
    df["return_14d"] = ret["return_14d"]

    # Log return (for volatility modeling); clip ratio for stability
    pc = df["close"].astype(float)
    prev_c = pc.shift(1)
    lr = np.log(np.clip(pc / prev_c.replace(0, np.nan), 1e-12, None))
    df["log_return_1d"] = lr
    # EWMA volatility of log returns
    df["ewma_vol_logret_14"] = lr.ewm(span=VOLATILITY_ROLLING, adjust=False).std()
    df["ewma_vol_logret_30"] = lr.ewm(span=30, adjust=False).std()
    df["rolling_std_logret_14"] = lr.rolling(window=VOLATILITY_ROLLING, min_periods=2).std()
    df["rolling_std_logret_30"] = lr.rolling(window=30, min_periods=3).std()
    df["vol_regime_encoded"] = _vol_regime_from_ewma(df["ewma_vol_logret_14"])

    # Momentum: rolling sum of daily log returns
    df["momentum_logret_3d"] = lr.rolling(window=3, min_periods=1).sum()
    df["momentum_logret_7d"] = lr.rolling(window=7, min_periods=1).sum()
    df["momentum_logret_14d"] = lr.rolling(window=14, min_periods=1).sum()

    # ROC
    roc = add_rate_of_change(df, periods=ROC_PERIODS)
    for c in roc.columns:
        df[c] = roc[c]

    # ADX / trend strength
    adx_df = add_adx(df, period=ADX_PERIOD)
    for c in adx_df.columns:
        df[c] = adx_df[c]
    adx_col = f"adx_{ADX_PERIOD}"
    df["trend_strength_adx"] = (df[adx_col].astype(float) / 100.0).clip(0.0, 1.0)

    # EMA cross (normalized by price)
    df["ema_cross_norm"] = ((df["ema_20"] - df["ema_50"]) / (pc.replace(0, np.nan) + 1e-30)).clip(-0.5, 0.5)

    # Acceleration of log returns
    df["log_return_accel"] = lr.diff()

    # Feature interactions (tree-friendly nonlinear cues)
    r7 = df["return_7d"].astype(float).clip(-0.5, 0.5)
    rstd14 = df["rolling_std_logret_14"].astype(float).fillna(0.0).clip(0.0, 0.25)
    rsi_n = (df["rsi_14"].astype(float) / 100.0 - 0.5).clip(-0.5, 0.5)
    adx_n = (df[adx_col].astype(float) / 100.0).clip(0.0, 1.0)
    mac_h = df["macd_histogram"].astype(float).clip(-1e6, 1e6)
    acc = df["log_return_accel"].astype(float).fillna(0.0).clip(-0.1, 0.1)
    df["interact_mom7_vol"] = (r7 * rstd14).clip(-0.15, 0.15)
    df["interact_rsi_adx"] = (rsi_n * adx_n).clip(-0.5, 0.5)
    df["interact_macd_accel"] = (mac_h * acc * 100.0).clip(-2.0, 2.0)

    # Level in log space — stable magnitude for micro-cap coins (replaces raw close lags)
    df["log_close"] = np.log(np.maximum(pc, 1e-30))

    # Volume change
    df["volume_change"] = add_volume_change(df)

    # Lag features: log_close (not raw close) + daily_return
    for col, periods in [("log_close", LAG_PERIODS), ("daily_return", LAG_PERIODS)]:
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
    adx_col = f"adx_{ADX_PERIOD}"
    di_p = f"plus_di_{ADX_PERIOD}"
    di_m = f"minus_di_{ADX_PERIOD}"
    roc_cols = [f"roc_{n}" for n in ROC_PERIODS]
    base = [
        "rsi_14",
        "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower",
        "ema_20", "ema_50", "sma_20", "sma_50",
        "atr_14",
        "rolling_volatility_14",
        "daily_return", "return_3d", "return_7d", "return_14d",
        "log_return_1d",
        "ewma_vol_logret_14",
        "ewma_vol_logret_30",
        "rolling_std_logret_14",
        "rolling_std_logret_30",
        "vol_regime_encoded",
        "momentum_logret_3d",
        "momentum_logret_7d",
        "momentum_logret_14d",
        *roc_cols,
        adx_col,
        di_p,
        di_m,
        "trend_strength_adx",
        "ema_cross_norm",
        "log_return_accel",
        "interact_mom7_vol",
        "interact_rsi_adx",
        "interact_macd_accel",
        "log_close",
        "volume_change",
    ]
    for col in ["log_close", "daily_return"]:
        for p in LAG_PERIODS:
            base.append(f"{col}_lag_{p}")
    if include_targets:
        base.append(TARGET_LOG_RETURN_1D)
    return base
