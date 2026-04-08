"""
Detect flat / constant forecasts and degraded inputs (padding, schema drift).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.logging_setup import get_logger

logger = get_logger(__name__)

# Stable contract for path-level quality signals (recursive_forecast + API consumers).
_PATH_SIGNAL_DEFAULTS: dict[str, Any] = {
    "is_constant_prediction": False,
    "low_variance_warning": False,
    "degraded_input": False,
    "ensemble_log_return_var": 0.0,
    "price_relative_variance": 0.0,
    "model_horizon_variance_raw": {},
    "step0_raw_logret_by_model": {},
    "max_missing_feature_fill_ratio": 0.0,
}


def normalize_forecast_path_signals(partial: dict[str, Any] | None) -> dict[str, Any]:
    """
    Merge partial output from compute_path_output_signals with defaults.
    Prevents KeyError when an early-return branch omits keys.
    """
    merged = {**_PATH_SIGNAL_DEFAULTS, **(partial or {})}
    if not isinstance(merged.get("model_horizon_variance_raw"), dict):
        merged["model_horizon_variance_raw"] = {}
    if not isinstance(merged.get("step0_raw_logret_by_model"), dict):
        merged["step0_raw_logret_by_model"] = {}
    return merged


def compute_path_output_signals(
    path: list[dict[str, Any]],
    model_raw_by_step: dict[str, list[float]],
    *,
    max_missing_feature_fill_ratio: float,
    emergency_width_align_used: bool,
    spot_start: float,
) -> dict[str, Any]:
    """
    is_constant_prediction: ensemble path is flat in log-return and price space.
    low_variance_warning: suspiciously little movement but not fully flat.
    degraded_input: too many imputed zero features or bad align.
    """
    degraded_input = bool(
        max_missing_feature_fill_ratio > 0.32
        or (emergency_width_align_used and max_missing_feature_fill_ratio > 0.12)
    )

    if not path:
        return {
            "is_constant_prediction": True,
            "low_variance_warning": True,
            "degraded_input": degraded_input,
            "ensemble_log_return_var": 0.0,
            "price_relative_variance": 0.0,
            "model_horizon_variance_raw": {},
            "step0_raw_logret_by_model": {},
            "max_missing_feature_fill_ratio": float(max_missing_feature_fill_ratio),
        }

    ens_lr = [float(p["predicted_log_return"]) for p in path]
    var_lr = float(np.var(ens_lr)) if len(ens_lr) > 1 else 0.0
    pv = [float(p["predicted_price"]) for p in path]
    m = abs(float(np.mean(pv))) if pv else 0.0
    rel_var_p = float(np.var(pv)) / max(m * m, 1e-40)

    model_vars: dict[str, float] = {}
    for name, series in model_raw_by_step.items():
        if len(series) > 1:
            model_vars[name] = float(np.var(np.array(series, dtype=np.float64)))
        else:
            model_vars[name] = 0.0

    # Near-zero ensemble movement (log space)
    is_constant = (var_lr < 1e-14) and (rel_var_p < 1e-16)
    low_var = (var_lr < 1e-10) or (rel_var_p < 1e-12) or (float(np.ptp(ens_lr)) < 1e-9)

    if all(v < 1e-16 for v in model_vars.values()) and model_vars:
        is_constant = True
        low_var = True

    step0_raw = {}
    for name, series in model_raw_by_step.items():
        if series:
            step0_raw[name] = round(float(series[0]), 10)

    logger.info(
        "Forecast path stats (spot=%.12g): ensemble_log_var=%.2e price_rel_var=%.2e models_raw_var=%s degraded_input=%s",
        spot_start,
        var_lr,
        rel_var_p,
        {k: round(v, 12) for k, v in model_vars.items()},
        degraded_input,
    )

    return {
        "is_constant_prediction": bool(is_constant),
        "low_variance_warning": bool(low_var) and not is_constant,
        "degraded_input": degraded_input,
        "ensemble_log_return_var": var_lr,
        "price_relative_variance": rel_var_p,
        "model_horizon_variance_raw": model_vars,
        "step0_raw_logret_by_model": step0_raw,
        "max_missing_feature_fill_ratio": max_missing_feature_fill_ratio,
    }
