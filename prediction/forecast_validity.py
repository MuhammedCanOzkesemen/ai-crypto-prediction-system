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
    """Documented composition for product/engineering (mirrors forecast_intel weights + penalties)."""
    return (
        "confidence_score combines: (1) cross-model log-return variance — very low variance is penalized "
        "(clip-collapse / shared failure); (2) historical test MAE/RMSE; (3) volatility regime; "
        "(4) path agreement scores. Multiplicative penalties for: non-production artifact_mode, fallback_mode, "
        "missing exact feature match, cumulative realism guardrail, clip-saturation fraction, sanity outlier vs "
        "historical tail, constant path (cap 0.12), low-variance path, degraded_input / imputation. "
        "artifact_mode legacy/degraded applies extra multipliers in compute_financial_confidence(). "
        "Signal-strength score, directional-classifier agreement/uncertainty, and vol regime adjust confidence; "
        "finalize_forecast_confidence() caps when agreement is low and the classifier is uncertain; "
        "caps also in apply_confidence_penalty_caps(). "
        "See compute_financial_confidence() and predictor predict_from_latest_features() for parameters."
    )
