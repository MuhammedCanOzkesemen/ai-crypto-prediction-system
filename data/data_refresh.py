"""
Feature data freshness checks and automatic OHLCV refresh + feature rebuild.

Uses Binance-first fetch (via existing fetcher), feature_builder, and parquet output
aligned with scripts/train_pipeline.run_data_and_features.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from features.feature_builder import build_features

from .price_fetcher import fetch_historical_data
from utils.config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

STALE_AFTER_HOURS = 24.0
_REFRESH_LOCKS: dict[str, threading.Lock] = {}


def _freshness_from_iso(
    latest_iso: str | None,
    *,
    stale_after_hours: float = STALE_AFTER_HOURS,
) -> tuple[bool, int | None]:
    """Return (is_fresh, data_age_hours floor). Aligns with API 24h stale rule."""
    if not latest_iso or not str(latest_iso).strip():
        return False, None
    try:
        ts = pd.to_datetime(latest_iso)
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        now = datetime.now(timezone.utc)
        age_sec = max(0.0, (now - ts).total_seconds())
        age_hours = int(age_sec // 3600)
        is_fresh = age_sec <= stale_after_hours * 3600.0
        return is_fresh, age_hours
    except Exception:
        return False, None


def _coin_lock(coin: str) -> threading.Lock:
    if coin not in _REFRESH_LOCKS:
        _REFRESH_LOCKS[coin] = threading.Lock()
    return _REFRESH_LOCKS[coin]


def _features_path(coin: str, features_dir: Path | None = None) -> Path:
    fd = Path(features_dir) if features_dir else settings.training.features_dir
    return fd / f"{coin.replace(' ', '_')}_features.parquet"


def _refresh_meta_path(coin: str) -> Path:
    base = settings.training.features_dir.parent / "cache" / "data_refresh"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{coin.replace(' ', '_')}_last_refresh.json"


@dataclass(frozen=True)
class DataFreshness:
    """Result of check_data_freshness."""

    is_fresh: bool
    data_age_hours: int | None
    latest_market_timestamp: str | None


def _latest_timestamp_iso_from_parquet(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date"])
    except Exception:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.warning("Could not read feature parquet %s: %s", path, e)
            return None
    if df.empty or "date" not in df.columns:
        return None
    try:
        last = df["date"].max()
        ts = pd.to_datetime(last)
        return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    except Exception:
        return None


def check_data_freshness(
    coin: str,
    features_dir: Path | None = None,
    *,
    stale_after_hours: float = STALE_AFTER_HOURS,
) -> DataFreshness:
    """
    Load latest market date from feature parquet and compare to current UTC time.

    Returns is_fresh True when feature data is newer than stale_after_hours.
    """
    path = _features_path(coin, features_dir)
    latest_iso = _latest_timestamp_iso_from_parquet(path)
    if latest_iso is None:
        return DataFreshness(is_fresh=False, data_age_hours=None, latest_market_timestamp=None)

    is_fresh, age_hours = _freshness_from_iso(latest_iso, stale_after_hours=stale_after_hours)
    return DataFreshness(
        is_fresh=is_fresh,
        data_age_hours=age_hours,
        latest_market_timestamp=latest_iso,
    )


def get_last_refresh_time(coin: str) -> str | None:
    """ISO timestamp of last successful forced or auto refresh for this coin."""
    p = _refresh_meta_path(coin)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        t = data.get("last_refresh_time")
        return str(t) if t else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _write_last_refresh(coin: str) -> None:
    payload: dict[str, Any] = {
        "coin": coin,
        "last_refresh_time": datetime.now(timezone.utc).isoformat(),
    }
    p = _refresh_meta_path(coin)
    try:
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("Could not write refresh metadata for %s: %s", coin, e)


def _run_fetch_and_features(
    coin: str,
    *,
    days: int | None = None,
    features_dir: Path | None = None,
) -> bool:
    days = int(days if days is not None else settings.training.default_history_days)
    fd = Path(features_dir) if features_dir else settings.training.features_dir
    fd.mkdir(parents=True, exist_ok=True)

    price_df = fetch_historical_data(coin, days=days, source="binance")
    if price_df is None or price_df.empty or len(price_df) < 100:
        price_df = fetch_historical_data(coin, days=days, source=None)
    if price_df is None or price_df.empty:
        logger.error("Refresh failed: no OHLCV for %s", coin)
        return False

    feature_df = build_features(price_df, include_targets=True, coin=coin)
    if feature_df.empty:
        logger.error("Refresh failed: empty features for %s", coin)
        return False

    out_path = _features_path(coin, fd)
    try:
        feature_df.to_parquet(out_path, index=False)
        logger.info("Saved refreshed features for %s to %s", coin, out_path)
    except Exception as e:
        logger.exception("Failed to save features for %s: %s", coin, e)
        return False
    return True


def refresh_data_if_needed(
    coin: str,
    *,
    force: bool = False,
    features_dir: Path | None = None,
    days: int | None = None,
    stale_after_hours: float = STALE_AFTER_HOURS,
) -> bool:
    """
    If feature data is older than stale_after_hours (or missing), or force=True:
    fetch OHLCV, rebuild features, save parquet.

    Returns True if a refresh was performed and succeeded, False if skipped or failed.

    Logs: "Data is fresh" when skipping; "Refreshing data..." / "Refresh complete" when running.
    """
    with _coin_lock(coin):
        if not force:
            chk = check_data_freshness(coin, features_dir, stale_after_hours=stale_after_hours)
            if chk.is_fresh and _features_path(coin, features_dir).exists():
                logger.info("Data is fresh")
                return False

        logger.info("Refreshing data...")
        ok = _run_fetch_and_features(coin, days=days, features_dir=features_dir)
        if ok:
            logger.info("Refresh complete")
            _write_last_refresh(coin)
        return ok
