"""
Strict trade opportunity filter: fewer, higher-conviction long signals.

Produces STRONG_BUY / WEAK_BUY / NO_TRADE from forecast diagnostics.
Edge score = confidence × mean path agreement × signal strength (then optional modifiers).
"""

from __future__ import annotations

import math
from typing import Any


def compute_trade_engine_edge_score(
    confidence_score: float,
    mean_path_agreement: float,
    signal_strength_score: float,
) -> float:
    """Product in [0, 1] when inputs are in [0, 1]."""
    c = float(max(0.0, min(1.0, confidence_score)))
    a = float(max(0.0, min(1.0, mean_path_agreement)))
    s = float(max(0.0, min(1.0, signal_strength_score)))
    return float(round(c * a * s, 6))


def _classifier_uncertain(directional_probs: dict[str, float] | None) -> bool:
    if not directional_probs:
        return True
    pup = float(directional_probs.get("up", 0.0))
    pdn = float(directional_probs.get("down", 0.0))
    pneu = float(directional_probs.get("neutral", 0.0))
    mx = max(pup, pdn, pneu)
    if mx < 0.42:
        return True
    if abs(pup - pdn) < 0.06:
        return True
    return False


def _trend_bearish(trend_label: str | None) -> bool:
    t = (trend_label or "").strip().upper()
    return t in ("STRONG DOWN", "DOWN")


def _trend_bullish_ok(trend_label: str | None) -> bool:
    """Long-bias filter: exclude clear bearish path labels."""
    return not _trend_bearish(trend_label)


def evaluate_trade_opportunity(prediction_output: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate strict long-only trade tiers.

    Required keys (defaults applied if missing):
      confidence_score, mean_path_agreement, signal_strength_score,
      directional_probabilities, volatility_regime, expected_move_pct,
      risk_reward_ratio, trend_label (or ``trend``), forecast_bullish (bool).

    Optional: sentiment_alignment (-1/0/1), adx_14 (float),
      consensus_score, stability_score, trend_confirmation_score,
      volatility_shock_detected, feature_sanity_failed, signal_cooldown_active.
    """
    conf = float(prediction_output.get("confidence_score") or 0.0)
    agree = float(prediction_output.get("mean_path_agreement") or 0.0)
    sig = float(prediction_output.get("signal_strength_score") or 0.0)
    probs = prediction_output.get("directional_probabilities")
    if not isinstance(probs, dict):
        probs = {}
    vol = str(prediction_output.get("volatility_regime") or "MEDIUM").upper()
    if vol not in ("LOW", "MEDIUM", "HIGH", "UNKNOWN"):
        vol = "MEDIUM"
    em = float(prediction_output.get("expected_move_pct") or 0.0)
    rr = float(prediction_output.get("risk_reward_ratio") or 0.0)
    trend = prediction_output.get("trend_label") or prediction_output.get("trend") or "NEUTRAL"
    bullish = bool(prediction_output.get("forecast_bullish", False))

    consensus_score = float(prediction_output.get("consensus_score") or 0.5)
    stability_score = float(prediction_output.get("stability_score") or 0.5)
    trend_confirmation_score = float(prediction_output.get("trend_confirmation_score") or 0.5)
    shock = bool(prediction_output.get("volatility_shock_detected", False))
    sanity_fail = bool(prediction_output.get("feature_sanity_failed", False))
    cooldown = bool(prediction_output.get("signal_cooldown_active", False))

    sa_raw = prediction_output.get("sentiment_alignment", 0.0)
    try:
        sentiment_align = float(sa_raw)
    except (TypeError, ValueError):
        sentiment_align = 0.0
    try:
        adx = float(prediction_output.get("adx_14") or 0.0)
    except (TypeError, ValueError):
        adx = 0.0

    reasons: list[str] = []
    base_score = compute_trade_engine_edge_score(conf, agree, sig)
    score = base_score

    if cooldown:
        reasons.append("Signal cooldown: no new trades until window expires")
        return {"decision": "NO_TRADE", "score": float(round(score, 6)), "reasons": reasons}

    if sanity_fail:
        reasons.append("Feature sanity check failed — forced NO_TRADE")
        return {"decision": "NO_TRADE", "score": float(round(score * 0.85, 6)), "reasons": reasons}

    # Hard veto (critical)
    if conf < 0.35 or agree < 0.30:
        reasons.append("Hard filter: confidence < 0.35 or mean path agreement < 0.30")
        return {"decision": "NO_TRADE", "score": float(round(score, 6)), "reasons": reasons}

    # Optional modifiers (capped)
    if vol == "HIGH":
        score *= 0.88
        reasons.append("Penalty: HIGH volatility regime")
    if _classifier_uncertain(probs):
        score *= 0.90
        reasons.append("Penalty: classifier uncertain (flat or low conviction)")

    if sentiment_align >= 0.75 and bullish and float(probs.get("up", 0.0)) >= 0.45:
        score = min(1.0, score * 1.06)
        reasons.append("Boost: Twitter sentiment aligned with long bias")
    if adx > 25.0 and math.isfinite(adx):
        score = min(1.0, score * 1.04)
        reasons.append("Boost: ADX > 25 (trend strength)")

    score = float(max(0.0, min(1.0, round(score, 6))))

    pup = float(probs.get("up", 0.0))
    pdn = float(probs.get("down", 0.0))

    strong_numeric = (
        conf >= 0.55
        and agree >= 0.60
        and sig >= 0.50
        and pup >= 0.60
        and em >= 0.03
        and rr >= 1.5
        and bullish
        and _trend_bullish_ok(str(trend))
    )
    strong_layers = (
        consensus_score >= 0.70
        and stability_score >= 0.58
        and trend_confirmation_score >= 0.56
        and not shock
    )

    if strong_numeric and strong_layers:
        reasons.insert(0, "Strong long: numeric gates + consensus/stability/trend confirmation")
        return {"decision": "STRONG_BUY", "score": score, "reasons": reasons}

    if strong_numeric and not strong_layers:
        if shock:
            reasons.append("Downgrade: volatility shock — STRONG_BUY blocked")
        elif consensus_score < 0.70:
            reasons.append("Downgrade: multi-day consensus insufficient for STRONG_BUY")
        elif stability_score < 0.58:
            reasons.append("Downgrade: forecast path stability insufficient for STRONG_BUY")
        elif trend_confirmation_score < 0.56:
            reasons.append("Downgrade: trend confirmation below threshold for STRONG_BUY")

    weak_ok = (
        conf >= 0.45
        and agree >= 0.50
        and em >= 0.015
        and bullish
        and _trend_bullish_ok(str(trend))
        and pup >= pdn
        and stability_score >= 0.40
        and consensus_score >= 0.38
    )
    if weak_ok:
        reasons.insert(0, "Weak scalp: minimum quality bar met for long")
        return {"decision": "WEAK_BUY", "score": score, "reasons": reasons}

    reasons.append("No trade: strong or weak long criteria not satisfied")
    return {"decision": "NO_TRADE", "score": score, "reasons": reasons}
