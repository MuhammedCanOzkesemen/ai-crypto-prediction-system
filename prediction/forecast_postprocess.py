"""
Post-processing: smoothing, volatility-aware ranges, sanity checks, and path inertia.

Keeps forecast behavior realistic (no extreme day-to-day jumps) and API outputs stable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# Max single-day move in the forecast path (fraction of previous level)
MAX_DAILY_RETURN_ABS = 0.08
# Rolling average window after clamp (3-day)
SMOOTH_WINDOW = 3
# Minimum half-width as fraction of price (avoid fake precision)
_MIN_HALF_FRAC = 0.005
# Spread from models considered "wide" for confidence (fraction of mean)
_WIDE_SPREAD_FRAC = 0.12


def smooth_path_from_anchor(
    raw_prices: np.ndarray | list[float],
    anchor_close: float | None,
    *,
    max_daily_abs: float = MAX_DAILY_RETURN_ABS,
    ma_window: int = SMOOTH_WINDOW,
) -> np.ndarray:
    """
    Anchor path to spot, clamp each step to ±max_daily_abs return vs previous,
    then apply centered moving average (same length).
    If anchor_close is missing, only smoothing is applied (no step clamp from spot).
    """
    arr = np.array(raw_prices, dtype=np.float64).copy()
    if anchor_close is not None and float(anchor_close) > 0:
        prev = float(anchor_close)
        for i in range(len(arr)):
            lo, hi = prev * (1.0 - max_daily_abs), prev * (1.0 + max_daily_abs)
            arr[i] = float(np.clip(arr[i], lo, hi))
            prev = arr[i]

    if ma_window <= 1 or len(arr) < 2:
        return arr

    w = max(2, min(int(ma_window), len(arr)))
    smoothed = np.zeros_like(arr)
    half = w // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        smoothed[i] = float(np.mean(arr[lo:hi]))
    return smoothed


def _half_width_from_context(
    mid: float,
    day_index_1based: int,
    close: float | None,
    fc: dict[str, float] | None,
) -> float:
    """Volatility-scaled minimum half-width (ATR + rolling vol)."""
    c = float(close) if close and close > 0 else max(abs(float(mid)), 1.0)
    fc = fc or {}
    atr = float(fc.get("atr_14", 0) or 0)
    roll_vol = float(fc.get("rolling_volatility_14", 0.02) or 0.02)
    day_scale = float(day_index_1based) ** 0.5
    # High vol → wider bands
    vol_mult = 1.0 + min(2.0, roll_vol / 0.02) * 0.35
    atr_part = 1.5 * atr * day_scale * vol_mult if atr > 0 else 0.0
    vol_part = c * max(roll_vol, 1e-6) * day_scale * 1.25 * vol_mult
    min_frac = c * _MIN_HALF_FRAC
    return max(atr_part, vol_part, min_frac)


def recenter_and_widen_bounds(
    rows: list[dict[str, Any]],
    feature_context: dict[str, float] | None,
    current_close: float | None,
    mean_agreement: float,
) -> None:
    """
    In-place: set bounds around smoothed mid; expand if range too narrow or models disagree.
    """
    fc = feature_context or {}
    for row in rows:
        mid = float(row.get("predicted_price") or row.get("ensemble_prediction") or 0)
        d = int(row.get("day_index", 1))
        old_lo = float(row.get("lower_bound", mid))
        old_hi = float(row.get("upper_bound", mid))
        half_raw = max((old_hi - old_lo) / 2.0, abs(mid) * 0.004)
        half_ctx = _half_width_from_context(mid, d, current_close, fc)
        half = max(half_raw, half_ctx)
        # Heavy disagreement → more uncertainty
        if mean_agreement < 0.45:
            half *= 1.35
        elif mean_agreement < 0.55:
            half *= 1.15
        rel = 2.0 * half / max(abs(mid), 1e-12)
        if rel < _MIN_HALF_FRAC * 2:
            half = max(half, abs(mid) * _MIN_HALF_FRAC)
        row["lower_bound"] = round(mid - half, 6)
        row["upper_bound"] = round(mid + half, 6)


def scale_model_predictions(
    row: dict[str, Any],
    raw_mid: float,
    new_mid: float,
) -> None:
    """Scale per-model outputs so ensemble shape tracks smoothed path."""
    mp = row.get("model_predictions")
    if not isinstance(mp, dict) or not mp:
        return
    if raw_mid is None or abs(float(raw_mid)) < 1e-18:
        return
    ratio = float(new_mid) / float(raw_mid)
    row["model_predictions"] = {k: round(float(v) * ratio, 6) for k, v in mp.items()}


def apply_sanity_pass(
    rows: list[dict[str, Any]],
    mean_agreement: float,
) -> None:
    """Final pass: ensure lower < mid < upper and minimum separation."""
    for row in rows:
        mid = float(row.get("predicted_price") or row.get("ensemble_prediction") or 0)
        lo = float(row.get("lower_bound", mid))
        hi = float(row.get("upper_bound", mid))
        if lo >= hi:
            eps = max(abs(mid) * _MIN_HALF_FRAC, 1e-8)
            lo, hi = mid - eps, mid + eps
        if mid <= lo or mid >= hi:
            mid = (lo + hi) / 2.0
            row["predicted_price"] = round(mid, 6)
            row["ensemble_prediction"] = round(mid, 6)
        row["lower_bound"] = round(lo, 6)
        row["upper_bound"] = round(hi, 6)


def mean_relative_spread_final(rows: list[dict[str, Any]]) -> float:
    """Average (max-min)/mean of model preds on last day."""
    if not rows:
        return 0.0
    last = rows[-1]
    mp = last.get("model_predictions") or {}
    vals = [float(v) for v in mp.values()]
    if len(vals) < 2:
        return 0.0
    m = float(np.mean(np.abs(vals)))
    if m < 1e-12:
        return 1.0
    return float((max(vals) - min(vals)) / m)


def mean_path_model_relative_spread(rows: list[dict[str, Any]]) -> float:
    """Mean over days of (max-min)/|mean| for model forecasts."""
    spreads: list[float] = []
    for row in rows:
        mp = row.get("model_predictions") or {}
        vals = [float(v) for v in mp.values()]
        if len(vals) < 2:
            continue
        m = float(np.mean(np.abs(vals)))
        if m < 1e-12:
            continue
        spreads.append((max(vals) - min(vals)) / m)
    if not spreads:
        return 0.1
    return float(np.mean(spreads))


def market_data_freshness(
    latest_market_timestamp: str | None,
    *,
    stale_after_hours: float = 24.0,
) -> tuple[str, int | None, str | None]:
    """
    Returns (data_freshness 'fresh'|'stale', data_age_hours floor int, optional message).
    """
    if not latest_market_timestamp or not str(latest_market_timestamp).strip():
        return "stale", None, "Market data timestamp unknown."
    try:
        ts = pd.to_datetime(latest_market_timestamp)
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        now = datetime.now(timezone.utc)
        age_sec = max(0.0, (now - ts).total_seconds())
        age_hours = int(age_sec // 3600)
        if age_sec <= stale_after_hours * 3600.0:
            return "fresh", age_hours, None
        return "stale", age_hours, (
            f"Underlying market data is about {age_hours} hours old relative to model features."
        )
    except Exception:
        return "stale", None, "Could not determine data freshness."
