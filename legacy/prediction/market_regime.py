"""
Market regime classification from the latest feature snapshot and short history.

Labels: TRENDING, RANGING, VOLATILE — used for adaptive confidence and trade gating.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

REGIME_TRENDING = "TRENDING"
REGIME_RANGING = "RANGING"
REGIME_VOLATILE = "VOLATILE"


def detect_market_regime(features: dict[str, Any] | pd.Series | pd.DataFrame | None) -> dict[str, Any]:
    """
    Classify regime from feature context (dict/Series) or the last rows of a feature DataFrame.

    Returns:
        regime: "TRENDING" | "RANGING" | "VOLATILE"
        market_regime_confidence: float in [0, 1]
        detail: short diagnostic string
    """
    adx: float | None = None
    ewma_vol: float | None = None
    trend_consistency: float | None = None
    range_pos: float | None = None
    vol_series: list[float] = []

    if features is None:
        return {
            "regime": REGIME_RANGING,
            "market_regime_confidence": 0.35,
            "detail": "no_features",
        }

    if isinstance(features, pd.DataFrame):
        if features.empty:
            return {
                "regime": REGIME_RANGING,
                "market_regime_confidence": 0.35,
                "detail": "empty_frame",
            }
        last = features.iloc[-1]
        adx = _f(last.get("adx_14"))
        ewma_vol = _f(last.get("ewma_vol_logret_14"))
        trend_consistency = _f(last.get("trend_consistency_score"))
        range_pos = _f(last.get("range_position_14d_window"))
        if "ewma_vol_logret_14" in features.columns:
            vs = features["ewma_vol_logret_14"].dropna().astype(float).tail(24)
            vol_series = [float(x) for x in vs.tolist() if math.isfinite(float(x))]
    elif isinstance(features, pd.Series):
        adx = _f(features.get("adx_14"))
        ewma_vol = _f(features.get("ewma_vol_logret_14"))
        trend_consistency = _f(features.get("trend_consistency_score"))
        range_pos = _f(features.get("range_position_14d_window"))
    else:
        fc = dict(features)
        adx = _f(fc.get("adx_14"))
        ewma_vol = _f(fc.get("ewma_vol_logret_14"))
        trend_consistency = _f(fc.get("trend_consistency_score"))
        range_pos = _f(fc.get("range_position_14d_window"))

    adx = adx if adx is not None else 0.0
    ewma_vol = ewma_vol if ewma_vol is not None else 0.02
    tc = trend_consistency if trend_consistency is not None else 0.5
    rp = range_pos if range_pos is not None else 0.5

    # VOLATILE: vol spike vs recent local average
    volatile = False
    vol_conf = 0.5
    if len(vol_series) >= 8:
        cur = float(vol_series[-1])
        hist = [float(x) for x in vol_series[:-1] if float(x) > 0]
        if hist:
            m = float(np.mean(hist))
            if m > 1e-12:
                ratio = cur / m
                if ratio >= 1.75:
                    volatile = True
                    vol_conf = float(max(0.55, min(1.0, (ratio - 1.0) / 1.2)))

    if volatile:
        return {
            "regime": REGIME_VOLATILE,
            "market_regime_confidence": round(vol_conf, 4),
            "detail": "vol_spike_vs_recent",
        }

    # TRENDING: strong ADX + directional structure
    trending = adx > 25.0 and tc >= 0.48
    if trending:
        conf = float(
            max(
                0.0,
                min(
                    1.0,
                    0.42 + 0.28 * min(1.0, (adx - 25) / 30) + 0.30 * min(1.0, tc),
                ),
            )
        )
        return {
            "regime": REGIME_TRENDING,
            "market_regime_confidence": round(conf, 4),
            "detail": "adx_trend_consistency",
        }

    # RANGING: weak trend strength and price not at range extremes
    ranging = adx < 22.0 and 0.25 <= rp <= 0.75
    if ranging or adx < 20.0:
        conf = float(max(0.0, min(1.0, 0.85 - 0.02 * max(0.0, adx))))
        return {
            "regime": REGIME_RANGING,
            "market_regime_confidence": round(conf, 4),
            "detail": "low_adx_mid_range",
        }

    # Default: mild trend or transition → treat as ranging with moderate confidence
    return {
        "regime": REGIME_RANGING,
        "market_regime_confidence": 0.52,
        "detail": "default_transition",
    }


def _f(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None
