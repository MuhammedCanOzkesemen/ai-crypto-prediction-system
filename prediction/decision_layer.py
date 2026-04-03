"""
Trade decision layer: filters forecasts into actionable BUY / SELL / NO_TRADE signals.

Does not change model outputs — only gates, scores, and API-facing decision fields.
Default when uncertain: NO_TRADE.
"""

from __future__ import annotations

import math
from collections import deque
from threading import Lock
from typing import Any

_NO_TRADE_WINDOW = 96
_signal_history: dict[str, deque[int]] = {}
_hist_lock = Lock()

# ---------------------------------------------------------------------------
# Thresholds (conservative; tune without touching models)
# ---------------------------------------------------------------------------

MIN_CONFIDENCE_TRADE = 0.46
MIN_MEAN_PATH_AGREEMENT_TRADE = 0.58
MIN_SIGNAL_STRENGTH_TRADE = 0.36
MIN_EDGE_SCORE_TRADE = 0.45
MIN_RISK_REWARD_RATIO = 0.28
MIN_RISK_REWARD_RATIO_HIGH_VOL = 0.38

# Expected move buckets (fraction of spot)
EXPECTED_MOVE_WEAK_MAX = 0.02
EXPECTED_MOVE_MODERATE_MAX = 0.05

CLASSIFIER_MARGIN = 0.06
PRED_FLAT_EPS = 0.00015


def record_trade_signal_observation(coin: str | None, trade_signal: str) -> None:
    """Rolling window of NO_TRADE vs trade for adaptive threshold nudging."""
    if not coin or not str(coin).strip():
        return
    key = str(coin).strip()
    with _hist_lock:
        d = _signal_history.setdefault(key, deque(maxlen=_NO_TRADE_WINDOW))
        d.append(1 if trade_signal == "NO_TRADE" else 0)


def recent_no_trade_fraction(coin: str | None) -> float:
    if not coin or not str(coin).strip():
        return 0.0
    key = str(coin).strip()
    with _hist_lock:
        d = _signal_history.get(key)
        if not d or len(d) < 10:
            return 0.0
        return float(sum(d)) / float(len(d))


def adaptive_threshold_scale(
    volatility_regime: str,
    trend_consistency: float,
    no_trade_fraction: float,
) -> float:
    """
    Scale > 1 → stricter gates; < 1 → slightly relaxed (only when environment warrants).

    HIGH vol: tighter. LOW / stable trend consistency: looser. Persistent NO_TRADE + strong trend: tiny extra relief.
    """
    vr = str(volatility_regime or "MEDIUM").upper()
    if vr not in ("LOW", "MEDIUM", "HIGH"):
        vr = "MEDIUM"
    tc = float(max(0.0, min(1.0, trend_consistency)))
    s = 1.0
    if vr == "HIGH":
        s *= 1.05
    elif vr == "LOW":
        s *= 0.94
    if tc >= 0.66:
        s *= 0.93
    elif tc >= 0.52:
        s *= 0.97
    if no_trade_fraction > 0.82 and tc >= 0.58:
        s *= 0.97
    return float(max(0.86, min(1.12, s)))


def expected_move_pct(current_price: float, final_prediction: float) -> float:
    """Absolute relative move: |final - spot| / spot."""
    if current_price is None or float(current_price) <= 0:
        return 0.0
    return abs(float(final_prediction) - float(current_price)) / float(current_price)


def categorize_expected_move_strength(pct: float) -> str:
    if pct < EXPECTED_MOVE_WEAK_MAX:
        return "WEAK"
    if pct < EXPECTED_MOVE_MODERATE_MAX:
        return "MODERATE"
    return "STRONG"


def classifier_directional_confidence(directional_probs: dict[str, float] | None) -> float:
    """Max probability among up/down/neutral — how decisive the head is."""
    if not directional_probs:
        return 0.0
    p = {k: float(v) for k, v in directional_probs.items()}
    return float(max(p.get("up", 0.0), p.get("down", 0.0), p.get("neutral", 0.0)))


def trend_strength_normalized(feature_context: dict[str, float] | None) -> float:
    """0..1 from ADX (14) — same spirit as signal_strength ADX leg."""
    fc = feature_context or {}
    adx = float(fc.get("adx_14") or 0.0)
    if adx <= 0:
        ts = float(fc.get("trend_strength_adx") or 0.0)
        if ts > 0:
            adx = ts * 100.0
    return float(max(0.0, min(1.0, (adx / 55.0) ** 0.88)))


def compute_edge_score(
    expected_move_pct_val: float,
    signal_strength_score: float,
    directional_probs: dict[str, float] | None,
    feature_context: dict[str, float] | None,
    *,
    trend_consistency_score: float | None = None,
    implied_return_vs_spot: float | None = None,
) -> float:
    """
    Normalized [0, 1] edge proxy: rewards coherent trend+momentum, not vol-only spikes.

    Base blend of implied move, signal strength, classifier, ADX; then refines using
    trend_consistency, vol expansion, and penalizes ``large move + weak structure``.
    """
    fc = feature_context or {}
    em = float(max(0.0, expected_move_pct_val))
    em_part = min(1.0, em / 0.08)
    ss = max(0.0, min(1.0, float(signal_strength_score)))
    clf = classifier_directional_confidence(directional_probs)
    if directional_probs and clf < 0.38:
        clf *= 0.84
    adx_n = trend_strength_normalized(fc)
    tc = float(trend_consistency_score) if trend_consistency_score is not None else float(fc.get("trend_consistency_score", 0.5) or 0.5)
    tc = max(0.0, min(1.0, tc))
    vol_ex = float(fc.get("vol_expansion_score") or 0.5)
    vol_ex = max(0.0, min(1.0, vol_ex))
    mom7 = abs(float(fc.get("return_7d") or 0.0))
    mom_sig = min(1.0, mom7 / 0.06)
    raw = (
        0.26 * em_part
        + 0.24 * ss
        + 0.20 * clf
        + 0.14 * adx_n
        + 0.10 * tc
        + 0.06 * mom_sig
    )
    edge = float(max(0.0, min(1.0, raw)))
    # Penalize edge that is mostly ``big implied move + vol expansion`` without trend support
    vol_only = em_part > 0.55 and vol_ex > 0.62 and tc < 0.40 and ss < 0.38
    if vol_only:
        edge *= 0.86
    # Reward coherent trend + momentum; penalize flat / uncommitted classifier distribution
    coherence = 0.52 * tc + 0.48 * max(ss, mom_sig)
    edge = min(1.0, edge * (0.76 + 0.24 * coherence))
    hint = float(implied_return_vs_spot) if implied_return_vs_spot is not None else None
    if hint is not None and math.isfinite(hint):
        r7 = float(fc.get("return_7d") or 0.0)
        if abs(hint) > 2e-5 and abs(r7) > 2e-5:
            if (hint > 0) == (r7 > 0):
                edge = min(1.0, edge * 1.05)
            else:
                edge *= 0.90
    return float(max(0.0, min(1.0, edge)))


def compute_risk_reward_ratio(
    current_price: float,
    final_prediction: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """
    reward = |predicted_move| in price space; risk = width of uncertainty band.

    risk_reward_ratio = reward / max(risk, epsilon)
    """
    reward = abs(float(final_prediction) - float(current_price))
    lo = float(lower_bound)
    hi = float(upper_bound)
    if lo > hi:
        lo, hi = hi, lo
    risk = hi - lo
    risk = max(risk, max(abs(float(current_price)) * 1e-8, 1e-12))
    return float(reward / risk)


def risk_reward_effective_for_gate(
    ratio: float,
    volatility_regime: str,
) -> float:
    """Penalize ratio in HIGH vol when comparing to threshold (stricter effective bar)."""
    r = float(max(0.0, ratio))
    if str(volatility_regime).upper() == "HIGH":
        return r * 0.82
    return r


def prediction_sign(current_price: float, final_prediction: float) -> int:
    if current_price <= 0:
        return 0
    rel = (float(final_prediction) - float(current_price)) / float(current_price)
    if rel > PRED_FLAT_EPS:
        return 1
    if rel < -PRED_FLAT_EPS:
        return -1
    return 0


def trend_sign_from_label(trend_label: str | None) -> int:
    t = (trend_label or "NEUTRAL").strip().upper()
    if t in ("STRONG UP", "UP"):
        return 1
    if t in ("STRONG DOWN", "DOWN"):
        return -1
    return 0


def classifier_agrees_with_sign(
    directional_probs: dict[str, float] | None,
    sign: int,
    margin: float = CLASSIFIER_MARGIN,
) -> bool:
    if sign == 0 or not directional_probs:
        return False
    pup = float(directional_probs.get("up", 0.0))
    pdn = float(directional_probs.get("down", 0.0))
    if sign > 0:
        return pup >= pdn + margin
    return pdn >= pup + margin


def compute_directional_alignment(
    current_price: float,
    final_prediction: float,
    trend_label: str | None,
    directional_probs: dict[str, float] | None,
) -> bool:
    """Trend direction matches prediction move AND classifier agrees with that sign."""
    ps = prediction_sign(current_price, final_prediction)
    ts = trend_sign_from_label(trend_label)
    if ps == 0 or ts == 0 or ps != ts:
        return False
    return classifier_agrees_with_sign(directional_probs, ps)


def apply_trade_aware_confidence(
    base_confidence: float,
    *,
    edge_score: float,
    mean_path_agreement: float,
    risk_reward_ratio: float,
    rr_threshold: float,
    volatility_regime: str,
    directional_alignment: bool,
    trade_valid: bool,
    expected_move_strength: str,
    trend_consistency_score: float = 0.5,
) -> float:
    """
    Pull confidence toward trade reliability: weak setups → lower displayed confidence.

    Uses the same gates' geometry without double-counting finalize_forecast_confidence.
    """
    bc = float(max(0.0, min(1.0, base_confidence)))
    e = float(max(0.0, min(1.0, edge_score)))
    a = float(max(0.0, min(1.0, mean_path_agreement)))
    rr_norm = float(max(0.0, min(1.0, risk_reward_ratio / max(rr_threshold * 2.2, 0.12))))
    vol_pen = 0.72 if str(volatility_regime).upper() == "HIGH" else 0.88 if str(volatility_regime).upper() == "MEDIUM" else 1.0
    al = 1.0 if directional_alignment else 0.62
    tv = 1.0 if trade_valid else 0.68
    em = 0.55 if expected_move_strength == "WEAK" else 0.82 if expected_move_strength == "MODERATE" else 1.0
    tc = float(max(0.0, min(1.0, trend_consistency_score)))
    rel = (
        0.18 * e
        + 0.18 * a
        + 0.16 * rr_norm
        + 0.13 * vol_pen
        + 0.13 * al
        + 0.09 * tv
        + 0.04 * em
        + 0.09 * tc
    )
    rel = float(max(0.38, min(1.0, rel)))
    out = bc * rel
    return float(max(0.0, min(1.0, out)))


def compute_high_conviction_decision(
    confidence_adjusted: float,
    mean_path_agreement: float,
    signal_strength_score: float,
    directional_alignment: bool,
    volatility_regime: str,
    directional_probs: dict[str, float] | None,
    prediction_sign_val: int,
) -> bool:
    """Strict bar: only True when trade-quality, alignment, and vol are favorable."""
    if str(volatility_regime).upper() == "HIGH":
        return False
    clf_ok = classifier_agrees_with_sign(directional_probs, prediction_sign_val) if prediction_sign_val != 0 else False
    return bool(
        confidence_adjusted >= 0.56
        and mean_path_agreement >= 0.72
        and signal_strength_score >= 0.46
        and directional_alignment
        and clf_ok
        and str(volatility_regime).upper() in ("LOW", "MEDIUM")
    )


def compute_decision_bundle(
    *,
    current_price: float,
    final_prediction: float,
    lower_bound: float,
    upper_bound: float,
    base_confidence: float,
    mean_path_agreement: float,
    signal_strength_score: float,
    trend_label: str,
    directional_probs: dict[str, float] | None,
    volatility_regime: str,
    feature_context: dict[str, float] | None,
    is_constant_prediction: bool = False,
    degraded_input: bool = False,
    low_variance_warning: bool = False,
    coin: str | None = None,
    trend_consistency_score: float | None = None,
) -> dict[str, Any]:
    """
    Full decision payload for API + predictor diagnostics.

    Trade signal defaults to NO_TRADE when any safety check fails.
    """
    reasons: list[str] = []
    cp = float(current_price)
    fp = float(final_prediction)
    lo = float(lower_bound)
    hi = float(upper_bound)
    fc = feature_context or {}
    tc_score = float(trend_consistency_score) if trend_consistency_score is not None else float(
        fc.get("trend_consistency_score") or 0.5
    )
    tc_score = max(0.0, min(1.0, tc_score))

    em_pct = expected_move_pct(cp, fp)
    em_strength = categorize_expected_move_strength(em_pct)
    edge = compute_edge_score(
        em_pct,
        signal_strength_score,
        directional_probs,
        feature_context,
        trend_consistency_score=tc_score,
        implied_return_vs_spot=((fp - cp) / cp) if cp > 0 else None,
    )
    rr_raw = compute_risk_reward_ratio(cp, fp, lo, hi)
    vr = str(volatility_regime or "MEDIUM").upper()
    if vr not in ("LOW", "MEDIUM", "HIGH"):
        vr = "MEDIUM"
    rr_eff = risk_reward_effective_for_gate(rr_raw, vr)
    nt_frac = recent_no_trade_fraction(coin)
    thr_scale = adaptive_threshold_scale(vr, tc_score, nt_frac)
    rr_base = MIN_RISK_REWARD_RATIO_HIGH_VOL if vr == "HIGH" else MIN_RISK_REWARD_RATIO
    rr_need = float(rr_base * thr_scale)
    edge_need = float((MIN_EDGE_SCORE_TRADE + (0.07 if vr == "HIGH" else 0.0)) * thr_scale)
    min_conf = float(MIN_CONFIDENCE_TRADE * thr_scale)
    min_agree = float(MIN_MEAN_PATH_AGREEMENT_TRADE * thr_scale)
    min_ss = float(MIN_SIGNAL_STRENGTH_TRADE * thr_scale)

    ps = prediction_sign(cp, fp)
    aligned = compute_directional_alignment(cp, fp, trend_label, directional_probs)
    clf_agrees = classifier_agrees_with_sign(directional_probs, ps) if ps != 0 else False

    trade_valid = True
    if is_constant_prediction:
        trade_valid = False
        reasons.append("constant or flat ensemble path")
    if degraded_input:
        trade_valid = False
        reasons.append("degraded or incomplete features")
    if low_variance_warning:
        trade_valid = False
        reasons.append("low path variance (elevated model risk)")
    if cp <= 0:
        trade_valid = False
        reasons.append("invalid spot price")
    if em_strength == "WEAK":
        trade_valid = False
        reasons.append("expected move is WEAK (<2%) — no meaningful edge")
    if base_confidence < min_conf:
        trade_valid = False
        reasons.append(f"confidence below adaptive floor ({min_conf:.0%}, scale={thr_scale:.2f})")
    if mean_path_agreement < min_agree:
        trade_valid = False
        reasons.append(f"mean model agreement below adaptive floor ({min_agree:.0%})")
    if signal_strength_score < min_ss:
        trade_valid = False
        reasons.append(f"signal strength below adaptive floor ({min_ss:.0%})")
    if edge < edge_need:
        trade_valid = False
        reasons.append(
            f"edge score below adaptive minimum (need ≥{edge_need:.2f}, got {edge:.2f}"
            + ("; HIGH vol regime" if vr == "HIGH" else "")
            + ")"
        )
    if rr_eff < rr_need:
        trade_valid = False
        reasons.append(
            f"risk-reward too low (effective {rr_eff:.3f} vs adaptive need {rr_need:.3f}"
            + ("; HIGH vol shrink applied to effective R:R" if vr == "HIGH" else "")
            + ")"
        )
    if ps == 0:
        trade_valid = False
        reasons.append("forecast is flat vs spot")
    if trend_sign_from_label(trend_label) == 0:
        trade_valid = False
        reasons.append("14d trend is NEUTRAL — no directional conviction")
    if not aligned:
        trade_valid = False
        reasons.append("trend, price path, and/or classifier not aligned")

    trade_signal = "NO_TRADE"
    if trade_valid and ps > 0:
        trade_signal = "BUY"
    elif trade_valid and ps < 0:
        trade_signal = "SELL"
    else:
        trade_signal = "NO_TRADE"

    record_trade_signal_observation(coin, trade_signal)

    conf_adj = apply_trade_aware_confidence(
        base_confidence,
        edge_score=edge,
        mean_path_agreement=mean_path_agreement,
        risk_reward_ratio=rr_raw,
        rr_threshold=rr_need,
        volatility_regime=vr,
        directional_alignment=aligned,
        trade_valid=(trade_signal != "NO_TRADE"),
        expected_move_strength=em_strength,
        trend_consistency_score=tc_score,
    )

    hc = compute_high_conviction_decision(
        conf_adj,
        mean_path_agreement,
        signal_strength_score,
        aligned,
        vr,
        directional_probs,
        ps,
    )

    missing_parts: list[str] = []
    if mean_path_agreement < min_agree:
        missing_parts.append(f"higher path agreement (≥{min_agree:.0%})")
    if edge < edge_need:
        missing_parts.append(f"stronger edge score (≥{edge_need:.2f})")
    if rr_eff < rr_need:
        missing_parts.append(f"better risk-reward (effective ≥{rr_need:.2f})")
    if not aligned:
        missing_parts.append("alignment of 14d trend, forecast direction, and classifier")
    if em_strength == "WEAK":
        missing_parts.append("larger expected move (≥2% to leave WEAK)")
    if base_confidence < min_conf:
        missing_parts.append(f"higher base confidence (≥{min_conf:.0%})")

    return {
        "trade_signal": trade_signal,
        "edge_score": round(edge, 4),
        "expected_move_pct": round(em_pct, 6),
        "expected_move_strength": em_strength,
        "risk_reward_ratio": round(rr_raw, 6),
        "trade_valid": bool(trade_signal != "NO_TRADE"),
        "directional_alignment": bool(aligned),
        "confidence_after_decision": round(conf_adj, 4),
        "high_conviction": hc,
        "decision_rejection_reasons": reasons,
        "classifier_agrees_with_direction": bool(clf_agrees),
        "risk_reward_threshold_used": round(rr_need, 4),
        "edge_score_threshold_used": round(edge_need, 4),
        "trend_consistency_score": round(tc_score, 4),
        "decision_threshold_scale": round(thr_scale, 4),
        "recent_no_trade_fraction": round(nt_frac, 4),
        "trade_missing_for_actionable": missing_parts[:6],
    }


def format_trade_decision_explanation(bundle: dict[str, Any], *, mean_path_agreement: float) -> str:
    """Actionable explanation: primary block reason + what is missing for a trade."""
    ts = str(bundle.get("trade_signal", "NO_TRADE"))
    tv = bool(bundle.get("trade_valid", False))
    em = str(bundle.get("expected_move_strength", ""))
    rr = bundle.get("risk_reward_ratio")
    edge = bundle.get("edge_score")
    reasons = bundle.get("decision_rejection_reasons") or []
    agree_pct = int(round(float(mean_path_agreement) * 100))
    tc = bundle.get("trend_consistency_score")
    scale = bundle.get("decision_threshold_scale")
    nt = bundle.get("recent_no_trade_fraction")
    missing = bundle.get("trade_missing_for_actionable") or []

    if tv and ts in ("BUY", "SELL"):
        return (
            f"Trade filter: {ts} is allowed — expected move {em}, risk-reward ≈{rr}, "
            f"edge score {edge}, mean path agreement {agree_pct}%, trend consistency {tc}. "
            f"Size and manage risk to your own policy; this is not financial advice."
        )
    top_fail = (
        " | ".join(f"{i + 1}) {r}" for i, r in enumerate(reasons[:4]))
        if reasons
        else "uncertainty or misalignment across models and filters"
    )
    miss_s = (
        " To reach a trade, typically need: " + "; ".join(missing[:4]) + "."
        if missing
        else ""
    )
    ctx = f" [trend_consistency={tc}, threshold_scale={scale}, recent_NO_TRADE_share≈{nt}]"
    return (
        f"Trade filter: NO_TRADE — failed checks: {top_fail}.{miss_s} "
        f"(Move: {em}, R:R≈{rr}, edge {edge}, agreement≈{agree_pct}%.){ctx}"
    )
