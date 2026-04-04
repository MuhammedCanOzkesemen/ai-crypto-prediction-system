"""
Final validity tier and composite quality score from path + artifact + diversity signals.
"""

from __future__ import annotations

from typing import Any


def compute_forecast_validity_and_quality_score(diag: dict[str, Any]) -> tuple[str, float]:
    """
    forecast_validity:
      - invalid: constant path or extreme breakdown
      - questionable: low variance, degraded input, guardrail, diversity collapse
      - valid: otherwise (still not a guarantee of profit)

    forecast_quality_score: 0..1 monotonic blend for dashboards (not confidence — use confidence_score separately).
    """
    if diag.get("is_constant_prediction"):
        return "invalid", 0.08
    if diag.get("artifact_mode") == "legacy":
        base = 0.22
    elif diag.get("artifact_mode") == "degraded":
        base = 0.48
    else:
        base = 0.82

    penalties = 0.0
    if diag.get("low_variance_warning"):
        penalties += 0.12
    if diag.get("degraded_input"):
        penalties += 0.14
    if diag.get("realism_guardrail_cumulative"):
        penalties += 0.08
    if diag.get("sanity_extreme"):
        penalties += 0.1
    if float(diag.get("clip_saturation_step_fraction", 0.0)) > 0.35:
        penalties += 0.1

    q = max(0.0, min(1.0, base - penalties))

    if q < 0.25:
        validity = "invalid"
    elif q < 0.55 or diag.get("low_variance_warning") or diag.get("degraded_input"):
        validity = "questionable"
    else:
        validity = "valid"

    if diag.get("is_constant_prediction"):
        validity = "invalid"

    return validity, round(q, 4)


def confidence_composition_doc() -> str:
    """Documented composition for product/engineering (hybrid directional + path stack)."""
    return (
        "Primary confidence_score is hybrid: 0.4×directional_confidence (max of merged UP/DOWN/NEUTRAL probs) + "
        "0.3×combined_agreement_score (regression path agreement blended with LR vs XGB directional agreement) + "
        "0.3×signal_strength_score. Light caps apply (HIGH vol, shock, chaotic disagreement, regime) via "
        "compose_light_risk_adjusted_confidence — Twitter/regime do not multiplicatively stack on top. "
        "legacy_financial_confidence_raw is retained as a diagnostic (old regression-heavy path). "
        "decision_layer may further adjust for trade UI; strict trade engine uses directional_confidence, "
        "combined_agreement_score, and expected_move_pct gates. See prediction/predictor.py and trade_filter.py."
    )
