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


def _dynamic_directional_threshold(
    volatility_regime: str,
    *,
    agreement_score: float,
    confidence_score: float,
) -> tuple[float, bool]:
    vr = str(volatility_regime or "MEDIUM").upper()
    if vr == "LOW":
        thr = 0.42
    elif vr == "HIGH":
        thr = 0.58
    else:
        thr = 0.48
    relaxed = False
    if agreement_score > 0.60 and confidence_score > 0.40:
        thr = max(0.36, thr - 0.05)
        relaxed = True
    return float(thr), relaxed


def _decision_summary(
    *,
    decision: str,
    reasons: list[str],
    blockers: dict[str, bool],
    key_drivers: list[str],
) -> dict[str, Any]:
    trade_reason = ""
    reject_reason = ""
    if decision != "NO_TRADE":
        trade_reason = reasons[0] if reasons else "Trade conditions met."
    else:
        reject_reason = reasons[0] if reasons else "Trade conditions not satisfied."
    return {
        "reason_for_trade": trade_reason,
        "reason_for_rejection": reject_reason,
        "key_drivers": key_drivers[:6],
        "decision_blockers": blockers,
    }


def evaluate_trade_opportunity(prediction_output: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate strict long-only trade tiers.

    Required keys (defaults applied if missing):
      confidence_score, mean_path_agreement, signal_strength_score,
      directional_probabilities, volatility_regime, expected_move_pct,
      risk_reward_ratio, trend_label (or ``trend``), forecast_bullish (bool).

    Optional: sentiment_alignment (-1/0/1), adx_14 (float),
      consensus_score, stability_score, trend_confirmation_score,
      volatility_shock_detected, feature_sanity_failed, signal_cooldown_active,
      market_regime (TRENDING|RANGING|VOLATILE), market_regime_confidence.
    """
    market_regime = str(prediction_output.get("market_regime") or "RANGING").strip().upper()
    if market_regime not in ("TRENDING", "RANGING", "VOLATILE"):
        market_regime = "RANGING"

    conf = float(prediction_output.get("confidence_score") or 0.0)
    base_conf = float(prediction_output.get("base_confidence") or conf)
    risk_adj_conf = float(prediction_output.get("risk_adjusted_confidence") or conf)
    agree = float(prediction_output.get("mean_path_agreement") or 0.0)
    comb_agree = float(prediction_output.get("combined_agreement_score") or agree)
    dir_conf = float(prediction_output.get("directional_confidence") or 0.0)
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
    agreement_trend_score = float(prediction_output.get("agreement_trend_score") or 0.5)
    agreement_trend_label = str(prediction_output.get("agreement_trend_label") or "mixed")

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
    key_drivers: list[str] = []
    base_score = compute_trade_engine_edge_score(conf, comb_agree, sig)
    score = base_score
    dir_thr, dir_relaxed = _dynamic_directional_threshold(
        vol,
        agreement_score=comb_agree,
        confidence_score=conf,
    )
    blockers = {
        "directional": bool(dir_conf <= dir_thr),
        "confidence": bool(conf <= 0.30),
        "agreement": bool(comb_agree <= 0.45),
        "risk_reward": bool(em <= 0.025 or rr < 1.05),
    }

    if comb_agree > 0.65:
        key_drivers.append("agreement strong")
    if agreement_trend_label in ("stable", "improving") and agreement_trend_score >= 0.55:
        key_drivers.append("agreement stable")
    if agreement_trend_label == "collapsing":
        key_drivers.append("agreement collapsing")
    if dir_conf > dir_thr:
        key_drivers.append("direction decisive")
    else:
        key_drivers.append("direction weak")
    if shock or vol == "HIGH":
        key_drivers.append("volatility high")
    if em >= 0.025:
        key_drivers.append("expected move actionable")
    if base_conf > risk_adj_conf + 0.06:
        key_drivers.append("risk penalties active")

    if cooldown:
        reasons.append("Signal cooldown: no new trades until window expires")
        return {
            "decision": "NO_TRADE",
            "score": float(round(score, 6)),
            "reasons": reasons,
            "decision_blockers": blockers,
            "decision_summary": _decision_summary(
                decision="NO_TRADE", reasons=reasons, blockers=blockers, key_drivers=key_drivers
            ),
        }

    if sanity_fail:
        reasons.append("Feature sanity check failed — forced NO_TRADE")
        return {
            "decision": "NO_TRADE",
            "score": float(round(score * 0.85, 6)),
            "reasons": reasons,
            "decision_blockers": blockers,
            "decision_summary": _decision_summary(
                decision="NO_TRADE", reasons=reasons, blockers=blockers, key_drivers=key_drivers
            ),
        }

    if conf < 0.22:
        blockers["confidence"] = True
        reasons.append("Hard filter: decision confidence < 0.22")
        return {
            "decision": "NO_TRADE",
            "score": float(round(score, 6)),
            "reasons": reasons,
            "decision_blockers": blockers,
            "decision_summary": _decision_summary(
                decision="NO_TRADE", reasons=reasons, blockers=blockers, key_drivers=key_drivers
            ),
        }
    if comb_agree < 0.40:
        blockers["agreement"] = True
        reasons.append("Hard filter: combined agreement < 0.40")
        return {
            "decision": "NO_TRADE",
            "score": float(round(score, 6)),
            "reasons": reasons,
            "decision_blockers": blockers,
            "decision_summary": _decision_summary(
                decision="NO_TRADE", reasons=reasons, blockers=blockers, key_drivers=key_drivers
            ),
        }

    if dir_conf <= dir_thr:
        reasons.append(
            f"Directional gate: confidence {dir_conf:.2f} below adaptive threshold {dir_thr:.2f}"
            + (" (relaxed by agreement/confidence)" if dir_relaxed else "")
        )
        return {
            "decision": "NO_TRADE",
            "score": float(round(score, 6)),
            "reasons": reasons,
            "decision_blockers": blockers,
            "decision_summary": _decision_summary(
                decision="NO_TRADE", reasons=reasons, blockers=blockers, key_drivers=key_drivers
            ),
        }
    if em <= 0.015:
        reasons.append("Expected move gate: horizon move must exceed 1.5%")
        return {
            "decision": "NO_TRADE",
            "score": float(round(score, 6)),
            "reasons": reasons,
            "decision_blockers": blockers,
            "decision_summary": _decision_summary(
                decision="NO_TRADE", reasons=reasons, blockers=blockers, key_drivers=key_drivers
            ),
        }

    if market_regime == "VOLATILE":
        score *= 0.7
        reasons.append("Regime VOLATILE: edge score scaled ×0.7")

    # Optional modifiers (capped)
    if vol == "HIGH":
        score *= 0.88
        reasons.append("Penalty: HIGH volatility regime")
    if _classifier_uncertain(probs):
        score *= 0.90
        reasons.append("Penalty: classifier uncertain (flat or low conviction)")
    if agreement_trend_label in ("stable", "improving") and agreement_trend_score >= 0.55:
        score = min(1.0, score * 1.05)
        reasons.append("Boost: agreement stayed stable across horizon")
    elif agreement_trend_label == "collapsing":
        score *= 0.90
        reasons.append("Penalty: agreement collapsed across horizon")

    if sentiment_align >= 0.75 and bullish and float(probs.get("up", 0.0)) >= 0.45:
        score = min(1.0, score * 1.06)
        reasons.append("Boost: Twitter sentiment aligned with long bias")
    if adx > 25.0 and math.isfinite(adx):
        score = min(1.0, score * 1.04)
        reasons.append("Boost: ADX > 25 (trend strength)")

    score = float(max(0.0, min(1.0, round(score, 6))))

    pup = float(probs.get("up", 0.0))
    pdn = float(probs.get("down", 0.0))

    volatile_extreme = (
        conf >= 0.72
        and comb_agree >= 0.78
        and sig >= 0.58
        and pup >= 0.62
        and stability_score >= 0.55
        and consensus_score >= 0.55
    )
    if market_regime == "VOLATILE" and not volatile_extreme:
        reasons.append("Regime VOLATILE: no trade without extreme confidence across core metrics")
        return {"decision": "NO_TRADE", "score": float(round(score, 6)), "reasons": reasons}

    agree_bar_strong = 0.58 if market_regime == "TRENDING" else 0.62
    strong_numeric = (
        conf >= 0.55
        and comb_agree >= agree_bar_strong
        and sig >= 0.50
        and pup >= 0.60
        and em >= 0.03
        and rr >= 1.5
        and bullish
        and _trend_bullish_ok(str(trend))
    )
    sc_need, st_need, tr_need = 0.70, 0.58, 0.56
    if market_regime == "TRENDING":
        sc_need, st_need, tr_need = 0.64, 0.54, 0.52
    strong_layers = (
        consensus_score >= sc_need
        and stability_score >= st_need
        and trend_confirmation_score >= tr_need
        and not shock
        and agreement_trend_label != "collapsing"
    )

    allow_strong = market_regime not in ("RANGING", "VOLATILE")
    if market_regime == "RANGING" and strong_numeric and strong_layers:
        reasons.append("Regime RANGING: STRONG_BUY blocked (range-bound context)")

    if allow_strong and strong_numeric and strong_layers:
        reasons.insert(0, "Strong long: numeric gates + consensus/stability/trend confirmation")
        if market_regime == "TRENDING":
            reasons.append("Regime TRENDING: slightly relaxed agreement / confirmation bars")
        return {"decision": "STRONG_BUY", "score": score, "reasons": reasons}

    if strong_numeric and not strong_layers:
        if shock:
            reasons.append("Downgrade: volatility shock — STRONG_BUY blocked")
        elif consensus_score < sc_need:
            reasons.append("Downgrade: multi-day consensus insufficient for STRONG_BUY")
        elif stability_score < st_need:
            reasons.append("Downgrade: forecast path stability insufficient for STRONG_BUY")
        elif trend_confirmation_score < tr_need:
            reasons.append("Downgrade: trend confirmation below threshold for STRONG_BUY")

    weak_ok = (
        conf >= 0.45
        and comb_agree >= 0.50
        and em >= 0.015
        and bullish
        and _trend_bullish_ok(str(trend))
        and pup >= pdn
        and stability_score >= 0.40
        and consensus_score >= 0.38
        and agreement_trend_label != "collapsing"
    )
    if market_regime == "RANGING":
        weak_ok = (
            weak_ok
            and conf >= 0.48
            and comb_agree >= 0.55
            and stability_score >= 0.48
            and consensus_score >= 0.45
            and em >= 0.018
        )
    if market_regime == "VOLATILE" and volatile_extreme:
        weak_ok = (
            weak_ok
            and em >= 0.025
            and rr >= 1.25
            and stability_score >= 0.50
        )

    if weak_ok:
        reasons.insert(0, "Weak scalp: minimum quality bar met for long")
        if market_regime == "RANGING":
            reasons.append("Regime RANGING: tight weak-only conditions")
        elif market_regime == "VOLATILE":
            reasons.append("Regime VOLATILE: weak entry only after extreme-confidence gate")
        return {
            "decision": "WEAK_BUY",
            "score": score,
            "reasons": reasons,
            "decision_blockers": blockers,
            "decision_summary": _decision_summary(
                decision="WEAK_BUY", reasons=reasons, blockers=blockers, key_drivers=key_drivers
            ),
        }

    probing_ok = (
        conf > 0.30
        and comb_agree > 0.45
        and em > 0.025
        and not shock
        and bullish
        and _trend_bullish_ok(str(trend))
        and dir_conf > dir_thr
        and pup >= pdn
        and sig >= 0.30
        and rr >= (1.15 if vol == "HIGH" else 1.05)
        and agreement_trend_label != "collapsing"
    )
    if probing_ok:
        reasons.insert(0, "Probing long: controlled entry with acceptable confidence, agreement, and move")
        return {
            "decision": "PROBING_BUY",
            "score": score,
            "reasons": reasons,
            "decision_blockers": blockers,
            "decision_summary": _decision_summary(
                decision="PROBING_BUY", reasons=reasons, blockers=blockers, key_drivers=key_drivers
            ),
        }

    reasons.append("No trade: strong or weak long criteria not satisfied")
    return {
        "decision": "NO_TRADE",
        "score": score,
        "reasons": reasons,
        "decision_blockers": blockers,
        "decision_summary": _decision_summary(
            decision="NO_TRADE", reasons=reasons, blockers=blockers, key_drivers=key_drivers
        ),
    }
