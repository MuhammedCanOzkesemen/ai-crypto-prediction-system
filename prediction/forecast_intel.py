"""
Forecast intelligence: confidence, trend labels, volatility level, explanations,
and multi-horizon summaries derived from the forecast path and feature context.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Trend: z = predicted return / daily vol (rolling); thresholds in "sigmas"
TREND_Z_STRONG = 2.0
TREND_Z_MILD = 0.5


def mean_directional_accuracy(metrics_by_model: dict[str, dict[str, float]]) -> float:
    """Average directional accuracy across models in [0, 1]."""
    accs: list[float] = []
    for m in metrics_by_model.values():
        da = m.get("directional_accuracy")
        if da is not None:
            v = float(da)
            if 0.0 <= v <= 1.0:
                accs.append(v)
    if not accs:
        return 0.5
    return float(np.mean(accs))


CONFIDENCE_MIN = 0.2
CONFIDENCE_MAX = 0.85
# Rolling vol above this → confidence penalty (typical daily return std scale)
VOL_REF = 0.035
# Relative model spread (max-min)/|mean| above this → low confidence
SPREAD_REF = 0.14


def compute_confidence_score(
    path_agreement_scores: list[float],
    day14_model_predictions: dict[str, float],
    metrics_by_model: dict[str, dict[str, float]],
    *,
    rolling_volatility_14: float | None = None,
    mean_path_relative_spread: float | None = None,
) -> float:
    """
    Realistic confidence in [CONFIDENCE_MIN, CONFIDENCE_MAX].

    Higher when models agree, volatility is moderate, and prediction spread is tight.
    Deliberately capped below ~0.85 to avoid overconfident outputs.
    """
    agree = float(np.mean(path_agreement_scores)) if path_agreement_scores else 0.0
    agree = max(0.0, min(1.0, agree))

    vals = list(day14_model_predictions.values())
    if mean_path_relative_spread is not None:
        rel_spread = float(mean_path_relative_spread)
    elif len(vals) >= 2:
        arr = np.array(vals, dtype=np.float64)
        m = float(np.mean(np.abs(arr)))
        rel_spread = float((float(arr.max()) - float(arr.min())) / max(m, 1e-12))
    else:
        rel_spread = 0.08

    inv_spread = max(0.0, 1.0 - min(1.0, rel_spread / SPREAD_REF))

    rv = float(rolling_volatility_14) if rolling_volatility_14 is not None else 0.025
    rv = max(rv, 1e-8)
    inv_vol = max(0.0, 1.0 - min(1.0, rv / VOL_REF))

    hist = mean_directional_accuracy(metrics_by_model)
    hist_c = max(0.35, min(0.75, hist))

    raw = 0.42 * agree + 0.28 * inv_vol + 0.20 * inv_spread + 0.10 * hist_c
    raw = max(0.0, min(1.0, float(raw)))
    scaled = CONFIDENCE_MIN + raw * (CONFIDENCE_MAX - CONFIDENCE_MIN)
    return float(min(CONFIDENCE_MAX, max(CONFIDENCE_MIN, scaled)))


def _trend_score_from_inputs(
    current_price: float,
    price_horizon_14d: float,
    rolling_volatility_14: float | None,
    return_7d: float | None,
) -> float:
    """
    Continuous trend score (~ -2.5 strong down .. +2.5 strong up) for smoothing / inertia.
    Blends 14d implied return vs vol with 7d momentum.
    """
    if current_price is None or current_price <= 0:
        return 0.0
    ret = (float(price_horizon_14d) - float(current_price)) / float(current_price)
    vol = float(rolling_volatility_14) if rolling_volatility_14 is not None else 0.02
    vol = max(vol, 1e-8)
    z14 = ret / vol
    z7 = 0.0
    if return_7d is not None:
        z7 = float(return_7d) / vol * 0.45
    return float(np.clip(z14 * 0.65 + z7, -3.0, 3.0))


def _label_from_score(z: float) -> str:
    if z >= TREND_Z_STRONG:
        return "STRONG UP"
    if z >= TREND_Z_MILD:
        return "UP"
    if z <= -TREND_Z_STRONG:
        return "STRONG DOWN"
    if z <= -TREND_Z_MILD:
        return "DOWN"
    return "SIDEWAYS"


def classify_trend_label(
    current_price: float,
    price_horizon_14d: float,
    rolling_volatility_14: float | None,
    return_7d: float | None = None,
    *,
    previous_trend_score: float | None = None,
    inertia_beta: float = 0.52,
) -> tuple[str, float]:
    """
    STRONG UP / UP / SIDEWAYS / DOWN / STRONG DOWN from predicted return,
    recent 7d return, and volatility. Blends with previous_trend_score for stability.
    Returns (label, effective_score).
    """
    z_new = _trend_score_from_inputs(
        current_price, price_horizon_14d, rolling_volatility_14, return_7d,
    )
    if previous_trend_score is not None:
        b = max(0.35, min(0.72, float(inertia_beta)))
        z_eff = b * previous_trend_score + (1.0 - b) * z_new
    else:
        z_eff = z_new
    return _label_from_score(z_eff), float(z_eff)


def classify_volatility_level(
    rolling_volatility_14: float | None,
    close: float | None,
) -> str:
    """Coarse market volatility bucket from rolling return volatility."""
    rv = float(rolling_volatility_14) if rolling_volatility_14 is not None else None
    if rv is None:
        return "UNKNOWN"
    # Typical daily return std: <1.5% low, <4% medium, else high
    if rv < 0.015:
        return "LOW"
    if rv < 0.04:
        return "MEDIUM"
    return "HIGH"


def build_prediction_explanation(
    feature_context: dict[str, float] | None,
    trend_label: str,
    confidence_score: float,
    mean_path_agreement: float,
    model_weights: dict[str, float],
) -> str:
    """
    Short, human-readable rationale (single paragraph). Rule-based from indicators
    plus model agreement / confidence.
    """
    parts: list[str] = []
    fc = feature_context or {}

    rsi = fc.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            parts.append("RSI is in oversold territory, which often precedes mean-reversion or relief rallies")
        elif rsi > 70:
            parts.append("RSI is elevated, suggesting stretched upside and higher pullback risk")

    ret7 = fc.get("return_7d")
    if ret7 is not None:
        if ret7 > 0.02:
            parts.append("seven-day momentum is positive")
        elif ret7 < -0.02:
            parts.append("seven-day momentum is negative")

    macd_h = fc.get("macd_histogram")
    if macd_h is not None:
        if macd_h > 0:
            parts.append("MACD histogram is bullish (momentum building)")
        elif macd_h < 0:
            parts.append("MACD histogram is bearish (momentum fading)")

    if not parts:
        parts.append("the ensemble reads current technical conditions as mixed")

    top_models = sorted(model_weights.items(), key=lambda x: -x[1])[:2]
    if top_models:
        top_s = ", ".join(f"{n.replace('_', ' ')} ({w * 100:.0f}%)" for n, w in top_models)
        parts.append(f"the weighted ensemble leans most on {top_s}")

    if mean_path_agreement < 0.55:
        parts.append("models disagree meaningfully day-to-day, so uncertainty is elevated")
    elif mean_path_agreement > 0.8:
        parts.append("models largely agree across the path")

    if confidence_score >= 0.58:
        parts.append("overall confidence in this outlook is moderately strong for this product band")
    elif confidence_score <= 0.32:
        parts.append("overall confidence is limited—treat ranges as wide")

    trend_phrase = {
        "STRONG UP": "The 14-day trajectory is strongly upward versus spot.",
        "UP": "The outlook is modestly upward over the two-week window.",
        "SIDEWAYS": "Price action is expected to be range-bound over the horizon.",
        "DOWN": "The outlook is modestly downward over the two-week window.",
        "STRONG DOWN": "The 14-day trajectory is strongly downward versus spot.",
    }.get(trend_label, "The trend classification is neutral.")

    core = "; ".join(parts[:5])
    if core:
        if not core.endswith("."):
            core += "."
        core = core[0].upper() + core[1:]
        return f"{trend_phrase} {core}"
    return trend_phrase


def build_multi_horizon(
    forecast_path: list[dict[str, Any]],
    current_price: float,
) -> dict[str, Any]:
    """Structured 1d / 3d / 7d / 14d snapshots from the 14-step path."""
    if not forecast_path:
        return {}
    by_day = {int(p["day_index"]): p for p in forecast_path if p.get("day_index") is not None}

    def snap(day_idx: int) -> dict[str, Any] | None:
        row = by_day.get(day_idx)
        if not row:
            return None
        pred = float(row.get("predicted_price") or row.get("ensemble_prediction") or 0)
        ret = None
        if current_price and current_price > 0:
            ret = (pred - current_price) / current_price
        return {
            "day_index": day_idx,
            "forecast_date": row.get("forecast_date"),
            "predicted_price": pred,
            "lower_bound": row.get("lower_bound"),
            "upper_bound": row.get("upper_bound"),
            "implied_return_vs_spot": round(ret, 6) if ret is not None else None,
            "agreement_score": row.get("agreement_score"),
        }

    out: dict[str, Any] = {}
    for label, idx in (("1d", 1), ("3d", 3), ("7d", 7), ("14d", 14)):
        s = snap(idx)
        if s:
            out[label] = s
    return out
