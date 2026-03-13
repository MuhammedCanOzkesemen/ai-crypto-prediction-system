"""
Historical cryptocurrency price fetcher.

Primary source: Binance API (long historical OHLCV, paginated).
Fallback: CoinGecko API.
Returns standardized DataFrame: date, open, high, low, close, volume.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from utils.constants import (
    COIN_DISPLAY_TO_ID,
    COIN_DISPLAY_TO_BINANCE,
    PRICE_COLUMNS,
    get_supported_coins_list,
)
from utils.config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Source constants
# -----------------------------------------------------------------------------

BINANCE_BASE_URL = "https://api.binance.com/api/v3"
BINANCE_KLINES_LIMIT = 1000  # max per request
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
STANDARD_COLUMNS = list(PRICE_COLUMNS)
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", "").strip() or None


def get_supported_coins() -> list[str]:
    """Return the list of supported coin display names."""
    return get_supported_coins_list()


def get_coin_list() -> list[str]:
    """Alias for get_supported_coins()."""
    return get_supported_coins()


def _coin_name_to_id(coin_name: str) -> str | None:
    """Resolve display name to CoinGecko id; case-insensitive."""
    if not coin_name or not coin_name.strip():
        return None
    key = coin_name.strip()
    for name, cid in COIN_DISPLAY_TO_ID.items():
        if name.lower() == key.lower():
            return cid
    return None


def _coin_name_to_binance(coin_name: str) -> str | None:
    """Resolve display name to Binance symbol; case-insensitive."""
    if not coin_name or not coin_name.strip():
        return None
    key = coin_name.strip()
    for name, sym in COIN_DISPLAY_TO_BINANCE.items():
        if name.lower() == key.lower():
            return sym
    return None


# -----------------------------------------------------------------------------
# Binance klines (paginated)
# -----------------------------------------------------------------------------


def _get_binance(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int | None = None,
) -> requests.Response | None:
    """Binance public API request (no auth)."""
    timeout = timeout or settings.data.request_timeout_sec
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        logger.warning("Binance request failed: %s", e)
        return None


def _fetch_klines_binance(symbol: str, days: int) -> pd.DataFrame | None:
    """
    Fetch OHLCV from Binance klines with pagination.
    Returns DataFrame with date, open, high, low, close, volume.
    """
    interval = "1d"
    end_ms = int(pd.Timestamp.now(tz="UTC").value // 1_000_000)
    start_ms = end_ms - (days + 2) * 24 * 60 * 60 * 1000  # +2 buffer
    all_rows: list[dict[str, Any]] = []

    while start_ms < end_ms:
        url = f"{BINANCE_BASE_URL}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": BINANCE_KLINES_LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        resp = _get_binance(url, params)
        if resp is None:
            break
        try:
            data = resp.json()
        except ValueError:
            break
        if not data or not isinstance(data, list):
            break
        for row in data:
            if not isinstance(row, (list, tuple)) or len(row) < 6:
                continue
            ts_ms = int(row[0])
            open_ = float(row[1])
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
            vol = float(row[5])
            all_rows.append({
                "date": pd.Timestamp(ts_ms, unit="ms").date(),
                "open": open_, "high": high, "low": low, "close": close, "volume": vol,
            })
        if len(data) < BINANCE_KLINES_LIMIT:
            break
        start_ms = int(data[-1][0]) + 1
        time.sleep(0.2)

    if not all_rows:
        return None
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[STANDARD_COLUMNS]


# -----------------------------------------------------------------------------
# CoinGecko (fallback)
# -----------------------------------------------------------------------------


def _get_coingecko(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int | None = None,
    retry_count: int = 0,
    max_retries_429: int = 4,
) -> requests.Response | None:
    """CoinGecko request with 401/429 handling."""
    timeout = timeout or settings.data.request_timeout_sec
    headers: dict[str, str] = {}
    if COINGECKO_API_KEY:
        headers["x-cg-demo-api-key"] = COINGECKO_API_KEY
    try:
        r = requests.get(url, params=params or {}, headers=headers or None, timeout=timeout)
        if r.status_code == 401:
            logger.error("CoinGecko 401 Unauthorized. Set COINGECKO_API_KEY env var.")
            return None
        if r.status_code == 429:
            if retry_count < max_retries_429:
                delay = 6 * (2**retry_count)
                logger.warning("CoinGecko rate limited (429); retry after %.0fs.", delay)
                time.sleep(delay)
                return _get_coingecko(url, params, timeout, retry_count + 1, max_retries_429)
            return None
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        logger.exception("CoinGecko request failed: %s", e)
        return None


def _fetch_ohlc_coingecko(coin_id: str, days: int) -> pd.DataFrame | None:
    """Fetch OHLC from CoinGecko (max 365 days)."""
    days_param = min(max(1, days), 365)
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": str(days_param)}
    resp = _get_coingecko(url, params)
    if resp is None:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not data or not isinstance(data, list):
        return None
    rows = []
    for row in data:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        ts_ms = int(row[0])
        open_, high, low, close = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        rows.append({"date": pd.Timestamp(ts_ms, unit="ms").date(), "open": open_, "high": high, "low": low, "close": close})
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["volume"] = 0.0
    return df[STANDARD_COLUMNS]


def _fetch_market_chart_coingecko(coin_id: str, days: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch market_chart (volume, market_cap)."""
    days_param = min(max(1, days), 365)
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days_param)}
    resp = _get_coingecko(url, params)
    if resp is None:
        return None, None
    try:
        data = resp.json()
    except ValueError:
        return None, None
    if not data or not isinstance(data, dict):
        return None, None

    def _series_to_df(key: str) -> pd.DataFrame | None:
        arr = data.get(key)
        if not arr or not isinstance(arr, list):
            return None
        rows = [{"date": pd.Timestamp(int(x[0]), unit="ms").date(), "value": float(x[1])} for x in arr if isinstance(x, (list, tuple)) and len(x) >= 2]
        if not rows:
            return None
        return pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return _series_to_df("total_volumes"), _series_to_df("market_caps")


def _fetch_coingecko(coin_id: str, days: int) -> pd.DataFrame | None:
    """Fetch OHLCV from CoinGecko (OHLC + volume merge)."""
    ohlc = _fetch_ohlc_coingecko(coin_id, days)
    if ohlc is None or ohlc.empty:
        return None
    time.sleep(settings.data.request_delay_sec)
    vol_df, _ = _fetch_market_chart_coingecko(coin_id, days)
    if vol_df is not None and not vol_df.empty:
        vol_df = vol_df.rename(columns={"value": "volume"})
        ohlc = ohlc.merge(vol_df[["date", "volume"]], on="date", how="left")
    ohlc["volume"] = ohlc["volume"].fillna(0.0) if "volume" in ohlc.columns else 0.0
    return ohlc[[c for c in STANDARD_COLUMNS if c in ohlc.columns]]


# -----------------------------------------------------------------------------
# Clean and public API
# -----------------------------------------------------------------------------


def _clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, sort by date, coerce numeric, drop invalid close."""
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0.0)
    return df


def fetch_historical_data(
    coin_name: str,
    days: int | None = None,
    source: str | None = None,
) -> pd.DataFrame | None:
    """
    Fetch historical OHLCV. Tries Binance first, falls back to CoinGecko.

    Parameters
    ----------
    coin_name : str
        Display name (e.g. "Bitcoin", "Ethereum"). Case-insensitive.
    days : int, optional
        Number of days of history. Default from config (e.g. 1500).
    source : str, optional
        "binance", "coingecko", or None (auto: Binance first, CoinGecko fallback).

    Returns
    -------
    pandas.DataFrame | None
        DataFrame with columns: date, open, high, low, close, volume.
    """
    days = days if days is not None else settings.data.default_history_days
    source = (source or "auto").lower()

    binance_sym = _coin_name_to_binance(coin_name)
    coin_id = _coin_name_to_id(coin_name)
    if coin_id is None and binance_sym is None:
        logger.warning("Unsupported coin: %s", coin_name)
        return None

    df: pd.DataFrame | None = None

    # Try Binance first (unless source=coingecko)
    if source != "coingecko" and binance_sym:
        df = _fetch_klines_binance(binance_sym, days)
        if df is not None and not df.empty and len(df) >= 100:
            logger.info("Fetched %s from Binance: %d rows", coin_name, len(df))
            return _clean_price_data(df)

    # Fallback to CoinGecko
    if coin_id and source != "binance":
        df = _fetch_coingecko(coin_id, min(days, 365))
        if df is not None and not df.empty:
            logger.info("Fetched %s from CoinGecko (fallback): %d rows", coin_name, len(df))
            return _clean_price_data(df)

    return None


def save_data_to_csv(
    coin_name: str,
    dataframe: pd.DataFrame,
    directory: str | Path | None = None,
) -> Path | None:
    """Save price DataFrame to CSV. Required columns: date, open, high, low, close, volume."""
    if dataframe is None or dataframe.empty:
        return None
    if not set(STANDARD_COLUMNS).issubset(set(dataframe.columns)):
        return None
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in coin_name.strip())
    filepath = directory / f"{safe_name}_prices.csv"
    try:
        dataframe.to_csv(filepath, index=False, date_format="%Y-%m-%d")
        return filepath
    except OSError:
        return None
