"""
Forecast intelligence: confidence, trend labels, volatility level, explanations,
and multi-horizon summaries derived from the forecast path and feature context.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from utils.constants import VOL_REGIME_LOGBOUND_HIGH, VOL_REGIME_LOGBOUND_LOW

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

# Reference scales for financial confidence (log-return variance / error / EWMA vol)
_FIN_VAR_REF = 0.00025
_FIN_ERR_REF = 0.05
_FIN_VOL_REF = 0.04


def apply_confidence_penalty_caps(confidence: float, diag: dict[str, Any]) -> float:
    """Post-process financial confidence from path-level honesty flags."""
    c = float(confidence)
    if diag.get("sanity_extreme"):
        c = min(c, 0.35)
    if diag.get("is_constant_prediction"):
        c = min(c, 0.12)
    elif diag.get("low_variance_warning"):
        c = min(c, 0.32)
    if diag.get("degraded_input"):
        c = min(c, 0.38)
    return float(max(0.0, min(1.0, c)))


def compute_signal_strength_score(
    current_price: float,
    final_price: float,
    feature_context: dict[str, float] | None,
) -> float:
    """
    0..1 composite: horizon move vs volatility, ADX, and 7d momentum magnitude.
    """
    fc = feature_context or {}
    if current_price is None or current_price <= 0:
        return 0.0
    ret = (float(final_price) - float(current_price)) / float(current_price)
    vol = max(float(fc.get("ewma_vol_logret_14") or 0.02), 1e-8)
    z_move = abs(ret) / max(vol * math.sqrt(14.0), 1e-8)
    adx_part = min(1.0, (float(fc.get("adx_14") or 0.0) / 45.0) ** 0.9)
    mom_part = min(1.0, abs(float(fc.get("return_7d") or 0.0)) / 0.06)
    raw = 0.42 * min(1.0, z_move / 2.2) + 0.33 * adx_part + 0.25 * mom_part
    return float(max(0.0, min(1.0, raw)))


def compute_high_conviction(
    *,
    signal_strength: float,
    mean_path_agreement: float,
    feature_context: dict[str, float] | None,
    implied_horizon_return: float,
    directional_probs: dict[str, float] | None,
    trend_label: str = "NEUTRAL",
) -> bool:
    """
    High conviction: strong ADX, meaningful 7d momentum, path agrees with momentum direction,
    models agree, signal strength sufficient, optional classifier not contradicting.
    Range-bound (NEUTRAL) 14d trend never qualifies as high conviction.
    """
    tl = (trend_label or "NEUTRAL").strip().upper()
    if tl == "NEUTRAL" or tl == "SIDEWAYS":
        return False
    fc = feature_context or {}
    adx = float(fc.get("adx_14") or 0.0)
    ts = adx / 100.0
    mom7 = float(fc.get("return_7d") or 0.0)
    strong_trend = ts >= 0.28
    strong_mom = abs(mom7) >= 0.018
    sign_r = 1 if implied_horizon_return > 0.004 else -1 if implied_horizon_return < -0.004 else 0
    sign_m = 1 if mom7 > 0.006 else -1 if mom7 < -0.006 else 0
    aligned = sign_r != 0 and sign_m == sign_r
    agree = mean_path_agreement >= 0.70
    clf_ok = True
    if directional_probs and sign_r != 0:
        pup = float(directional_probs.get("up", 0.0))
        pdn = float(directional_probs.get("down", 0.0))
        if sign_r > 0 and pup < pdn + 0.04:
            clf_ok = False
        if sign_r < 0 and pdn < pup + 0.04:
            clf_ok = False
    return bool(
        strong_trend
        and strong_mom
        and aligned
        and agree
        and signal_strength >= 0.40
        and clf_ok
    )


def volatility_regime_label(vol_regime_encoded: float | None) -> str:
    """Map encoded 0/1/2 to LOW/MEDIUM/HIGH (fallback MEDIUM if unknown)."""
    if vol_regime_encoded is None:
        return "MEDIUM"
    try:
        v = int(round(float(vol_regime_encoded)))
    except (TypeError, ValueError):
        return "MEDIUM"
    if v <= 0:
        return "LOW"
    if v == 1:
        return "MEDIUM"
    return "HIGH"


def infer_volatility_regime_from_context(feature_context: dict[str, float] | None) -> str:
    """
    Always returns LOW | MEDIUM | HIGH using vol_regime_encoded, then log-vol features,
    then rolling return volatility.
    """
    fc = feature_context or {}
    vre = fc.get("vol_regime_encoded")
    if vre is not None:
        try:
            xf = float(vre)
            if math.isfinite(xf):
                return volatility_regime_label(xf)
        except (TypeError, ValueError):
            pass
    for key in ("ewma_vol_logret_30", "rolling_std_logret_14", "ewma_vol_logret_14"):
        x = fc.get(key)
        if x is None:
            continue
        try:
            xv = float(x)
            if math.isfinite(xv) and xv >= 0:
                if xv < VOL_REGIME_LOGBOUND_LOW:
                    return "LOW"
                if xv < VOL_REGIME_LOGBOUND_HIGH:
                    return "MEDIUM"
                return "HIGH"
        except (TypeError, ValueError):
            continue
    rv = fc.get("rolling_volatility_14")
    if rv is not None:
        try:
            r = float(rv)
            if math.isfinite(r) and r >= 0:
                if r < 0.015:
                    return "LOW"
                if r < 0.04:
                    return "MEDIUM"
                return "HIGH"
        except (TypeError, ValueError):
            pass
    return "MEDIUM"


def _trend_label_from_z(z: float) -> str:
    """Discrete 14d trend from standardized score (NEUTRAL = range-bound)."""
    if z >= TREND_Z_STRONG:
        return "STRONG UP"
    if z >= TREND_Z_MILD:
        return "UP"
    if z <= -TREND_Z_STRONG:
        return "STRONG DOWN"
    if z <= -TREND_Z_MILD:
        return "DOWN"
    return "NEUTRAL"


def classify_trend_from_forecast_path(
    spot_close: float,
    path_rows: list[dict[str, Any]],
    *,
    log_vol_daily: float | None = None,
) -> tuple[str, float]:
    """
    14d trend from **unrounded** ensemble path: cumulative log return vs spot and average
    daily log drift, scaled by a log-return volatility prior (EWMA/rolling).
    """
    if spot_close <= 0 or not path_rows:
        return "NEUTRAL", 0.0
    final_p = float(path_rows[-1]["predicted_price"])
    cum_log = math.log(max(final_p, 1e-30) / max(float(spot_close), 1e-30))
    n = len(path_rows)
    v = max(float(log_vol_daily if log_vol_daily is not None else 0.018), 1e-6)
    z_cum = cum_log / max(v * math.sqrt(float(n)), 1e-10)
    avg_step = cum_log / float(n)
    z_slope = avg_step / max(v, 1e-10)
    z_eff = float(np.clip(0.58 * z_cum + 0.42 * z_slope, -3.5, 3.5))
    return _trend_label_from_z(z_eff), z_eff


def classifier_uncertainty_score(directional_probs: dict[str, float] | None) -> float:
    """0 = firm directional view, 1 = very uncertain (missing classifier treated as uncertain)."""
    if not directional_probs:
        return 0.62
    p = {k: float(v) for k, v in directional_probs.items()}
    m = max(p.get("up", 0.0), p.get("down", 0.0), p.get("neutral", 0.0))
    neu = p.get("neutral", 0.0)
    return float(max(0.0, min(1.0, (1.0 - m) + 0.35 * neu)))


def finalize_forecast_confidence(
    confidence: float,
    *,
    mean_path_agreement: float,
    directional_probs: dict[str, float] | None,
    vol_regime: str,
) -> float:
    """
    Enforce consistency: high disagreement + uncertain classifier caps confidence;
    HIGH vol regime trims tail confidence.
    """
    c = float(max(0.0, min(1.0, confidence)))
    agree = float(max(0.0, min(1.0, mean_path_agreement)))
    cu = classifier_uncertainty_score(directional_probs)

    if agree < 0.50:
        c *= 0.62 + 0.38 * (agree / 0.50)
    if agree < 0.42 and cu >= 0.48:
        c = min(c, 0.49)
    elif agree < 0.50 and cu >= 0.55:
        c = min(c, 0.52)
    if str(vol_regime).upper() == "HIGH":
        c *= 0.90
    return float(max(0.0, min(1.0, c)))


def compute_financial_confidence(
    path_agreement_scores: list[float],
    lr_matrix: list[list[float]],
    metrics_by_model: dict[str, dict[str, float]],
    forecast_vol_start: float,
    *,
    forecast_quality: str = "production",
    fallback_mode: bool = False,
    guardrail_cumulative: bool = False,
    clip_saturation_fraction: float = 0.0,
    exact_feature_match: bool = True,
    artifact_mode: str = "production",
    signal_strength: float = 0.0,
    directional_probs: dict[str, float] | None = None,
    implied_horizon_log_return: float = 0.0,
    vol_regime_encoded: float | None = None,
) -> float:
    """
    Confidence in [0, 1] from metrics, volatility, and model spread — heavily penalized
    when forecasts are degraded, guardrails fire, or outputs are clip-saturated together.
    """
    vars_per_step: list[float] = []
    for row in lr_matrix:
        if len(row) >= 2:
            vars_per_step.append(float(np.var(np.array(row, dtype=np.float64))))
    mean_var = float(np.mean(vars_per_step)) if vars_per_step else 0.0
    # Very low variance across models is suspicious (shared failure / clip collapse), not strength.
    if mean_var < 1e-8 and len(lr_matrix) >= 2:
        inv_model_var = 0.35
    else:
        inv_model_var = 1.0 / (1.0 + mean_var / max(_FIN_VAR_REF, 1e-12))
        inv_model_var = float(max(0.0, min(1.0, inv_model_var)))

    err_vals: list[float] = []
    for m in metrics_by_model.values():
        for k in ("mae", "rmse"):
            if m.get(k) is not None:
                e = float(m[k])
                if e > 1e-12:
                    err_vals.append(e)
                break
    mean_err = float(np.mean(err_vals)) if err_vals else _FIN_ERR_REF
    inv_error = 1.0 / (1.0 + mean_err / max(_FIN_ERR_REF, 1e-12))
    inv_error = float(max(0.0, min(1.0, inv_error)))

    v = max(float(forecast_vol_start), 1e-8)
    inv_volatility = 1.0 / (1.0 + v / max(_FIN_VOL_REF, 1e-12))
    inv_volatility = float(max(0.0, min(1.0, inv_volatility)))

    agree = float(np.mean(path_agreement_scores)) if path_agreement_scores else 0.5
    agree = max(0.0, min(1.0, agree))

    cu = classifier_uncertainty_score(directional_probs)

    # Agreement drives the blend weight — low agreement should not be masked by error/vol terms.
    agree_w = 0.22 if mean_var >= 1e-9 else 0.18
    w_var, w_err, w_vol = 0.26, 0.26, 0.26
    w_sum = w_var + w_err + w_vol + agree_w
    w_var, w_err, w_vol, agree_w = w_var / w_sum, w_err / w_sum, w_vol / w_sum, agree_w / w_sum

    combined = (
        w_var * inv_model_var
        + w_err * inv_error
        + w_vol * inv_volatility
        + agree_w * agree
    )
    combined = float(max(0.0, min(1.0, combined)))

    if agree < 0.50:
        combined *= 0.58 + 0.42 * (agree / 0.50)
    if cu >= 0.55:
        combined *= 0.88 + 0.12 * max(0.0, (1.0 - cu) / 0.45)

    ss = max(0.0, min(1.0, float(signal_strength)))
    if ss >= 0.50 and agree >= 0.68 and cu <= 0.48:
        combined = min(1.0, combined * (1.0 + 0.04 * (ss - 0.50) / 0.50))
    if (
        ss >= 0.55
        and agree >= 0.74
        and cu <= 0.42
        and vol_regime_encoded is not None
        and float(vol_regime_encoded) <= 1.25
    ):
        combined = min(1.0, combined * 1.025)

    if directional_probs:
        pup = float(directional_probs.get("up", 0.0))
        pdn = float(directional_probs.get("down", 0.0))
        pneu = float(directional_probs.get("neutral", 0.0))
        ilr = float(implied_horizon_log_return)
        if ilr > 0.0018 and pup >= 0.40 and pup >= pdn and agree >= 0.58:
            combined = min(1.0, combined * 1.022)
        elif ilr < -0.0018 and pdn >= 0.40 and pdn >= pup and agree >= 0.58:
            combined = min(1.0, combined * 1.022)
        elif pneu > 0.52 or max(pup, pdn, pneu) < 0.36:
            combined *= 0.90
        elif (ilr > 0.0025 and pdn > pup + 0.10) or (ilr < -0.0025 and pup > pdn + 0.10):
            combined *= 0.88

    if vol_regime_encoded is not None and float(vol_regime_encoded) >= 1.85:
        combined *= 0.86

    if forecast_quality != "production":
        combined *= 0.52
    if artifact_mode == "legacy":
        combined *= 0.48
    elif artifact_mode == "degraded":
        combined *= 0.88
    if fallback_mode:
        combined *= 0.68
    if not exact_feature_match:
        combined *= 0.72
    if guardrail_cumulative:
        combined *= 0.78
    if clip_saturation_fraction > 0.35:
        combined *= 0.62

    return float(max(0.0, min(1.0, combined)))


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
    return "NEUTRAL"


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
    STRONG UP / UP / NEUTRAL / DOWN / STRONG DOWN from predicted return,
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
    *,
    directional_probs: dict[str, float] | None = None,
    implied_horizon_log_return: float | None = None,
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

    if directional_probs:
        pup = float(directional_probs.get("up", 0.0))
        pdn = float(directional_probs.get("down", 0.0))
        pneu = float(directional_probs.get("neutral", 0.0))
        dom = max(pup, pdn, pneu)
        dom_l = "up" if dom == pup and pup >= pdn and pup >= pneu else (
            "down" if dom == pdn else "neutral"
        )
        parts.append(
            f"the directional classifier leans {dom_l} (≈{dom * 100:.0f}% probability vs other buckets)"
        )
        if implied_horizon_log_return is not None and math.isfinite(implied_horizon_log_return):
            ilr = float(implied_horizon_log_return)
            if ilr > 0.002 and pdn > pup + 0.08:
                parts.append("the classifier distribution conflicts with the upward 14d path—down-weight conviction")
            elif ilr < -0.002 and pup > pdn + 0.08:
                parts.append("the classifier distribution conflicts with the downward 14d path—down-weight conviction")

    if confidence_score >= 0.58:
        parts.append("overall confidence in this outlook is moderately strong for this product band")
    elif confidence_score <= 0.32:
        parts.append("overall confidence is limited—treat ranges as wide")

    trend_phrase = {
        "STRONG UP": "The 14-day trajectory is strongly upward versus spot.",
        "UP": "The outlook is modestly upward over the two-week window.",
        "NEUTRAL": "Price action is expected to be range-bound over the horizon.",
        "SIDEWAYS": "Price action is expected to be range-bound over the horizon.",
        "DOWN": "The outlook is modestly downward over the two-week window.",
        "STRONG DOWN": "The 14-day trajectory is strongly downward versus spot.",
    }.get(trend_label, "The 14-day trend versus spot is assessed as neutral.")

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
