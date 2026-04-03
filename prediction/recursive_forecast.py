"""
Recursive multi-step forecast: one-day log-return models, chained with price updates.

Risk-controlled daily caps (vol-scaled), cumulative compounding limit, and honesty
about model agreement (clip-collapse is not treated as high confidence).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from features.feature_builder import build_features
from prediction.forecast_path_quality import (
    compute_path_output_signals,
    normalize_forecast_path_signals,
)
from prediction.inference_utils import (
    align_X_to_n_features,
    classify_forecast_quality,
    exact_training_columns_present,
    feature_row_to_matrix_with_stats,
    infer_n_features_for_ensemble,
    prediction_matrix_to_dataframe,
    resolve_inference_columns,
)
from prediction.realism import (
    OUTER_DAILY_ABS_CAP,
    apply_cumulative_guardrail_to_path,
    clip_daily_log_return,
    daily_log_return_cap,
    model_output_diversity_agreement,
    summarize_path_sanity,
)
from utils.config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

HISTORY_TAIL = 1000
DEFAULT_HORIZON = 14

# Full diagnostics contract returned with every recursive forecast (defaults for missing keys).
_DEFAULT_RECURSIVE_DIAGNOSTICS: dict[str, Any] = {
    "forecast_quality": "degraded",
    "used_scaler": False,
    "exact_feature_match": False,
    "fallback_mode": True,
    "used_emergency_width_align": False,
    "scaler_transform_failed": False,
    "realism_guardrail_cumulative": False,
    "target_log_return_1d": "target_log_return_1d",
    "clip_saturation_step_fraction": 0.0,
    "training_metadata_present": False,
    "sanity_check": {},
    "sanity_extreme": False,
    "is_constant_prediction": False,
    "low_variance_warning": False,
    "degraded_input": False,
    "ensemble_log_return_var": 0.0,
    "price_relative_variance": 0.0,
    "model_horizon_variance_raw": {},
    "step0_raw_logret_by_model": {},
    "max_missing_feature_fill_ratio": 0.0,
}


def _normalize_recursive_forecast_diagnostics(d: dict[str, Any]) -> dict[str, Any]:
    out = {**_DEFAULT_RECURSIVE_DIAGNOSTICS, **d}
    if not isinstance(out.get("sanity_check"), dict):
        out["sanity_check"] = {}
    if not isinstance(out.get("model_horizon_variance_raw"), dict):
        out["model_horizon_variance_raw"] = {}
    if not isinstance(out.get("step0_raw_logret_by_model"), dict):
        out["step0_raw_logret_by_model"] = {}
    return out


def _weighted_log_return(preds: dict[str, float], weights: dict[str, float]) -> float:
    if not preds:
        return 0.0
    wsum = 0.0
    xsum = 0.0
    for name, r in preds.items():
        w = float(weights.get(name, 0.0))
        xsum += w * float(r)
        wsum += w
    if wsum < 1e-12:
        return float(np.mean(list(preds.values())))
    return float(xsum / wsum)


def recursive_log_return_forecast(
    history_ohlcv: pd.DataFrame,
    models: dict[str, Any],
    weights: dict[str, float],
    scaler_bundle: dict[str, Any] | None,
    *,
    training_meta: dict[str, Any] | None = None,
    horizon: int = DEFAULT_HORIZON,
    vol_k: float = 2.0,
) -> tuple[list[dict[str, Any]], dict[str, float], float, dict[str, Any]]:
    """
    Returns path rows, final model prices, ewma_vol at t0, and diagnostics for API/metadata.
    """
    cols = ["date", "open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in history_ohlcv.columns:
            raise ValueError(f"history_ohlcv missing column {c}")

    history = history_ohlcv[cols].copy().sort_values("date").reset_index(drop=True)
    feat_df0 = build_features(history.tail(min(len(history), HISTORY_TAIL)), include_targets=False)
    bundle = scaler_bundle if scaler_bundle is not None else {}
    scaler = bundle.get("scaler")
    saved_cols = bundle.get("feature_columns")
    columns = resolve_inference_columns(bundle, feat_df0.columns)
    if not columns:
        raise ValueError("No feature columns available for recursive forecast")

    meta = training_meta or {}
    p99_hist = meta.get("historical_abs_logret_p99")

    ewma_vol_start = 0.02
    path: list[dict[str, Any]] = []
    model_prices = {n: float(history["close"].iloc[-1]) for n in models}
    n_features_model = infer_n_features_for_ensemble(models)
    spot_start = float(history["close"].iloc[-1])

    emergency_align_used = False
    scaler_transform_failed_any = False
    saturation_steps = 0
    allow_align = settings.prediction.allow_emergency_feature_align
    max_miss_fill = 0.0
    model_raw_by_step: dict[str, list[float]] = {n: [] for n in models}

    for step in range(horizon):
        tail = history.tail(min(len(history), HISTORY_TAIL))
        feat_df = build_features(tail, include_targets=False)
        if feat_df.empty or len(feat_df) < 1:
            logger.warning("recursive_forecast: empty features at step %d", step)
            break
        last = feat_df.iloc[-1]
        try:
            X, row_stats = feature_row_to_matrix_with_stats(last, columns)
            max_miss_fill = max(max_miss_fill, float(row_stats["missing_feature_fill_ratio"]))
        except Exception as e:
            logger.warning("Feature row extract failed at step %d: %s", step, e)
            break
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if scaler is not None:
            try:
                Xs = scaler.transform(X)
            except Exception as e:
                logger.warning("Scaler transform failed: %s", e)
                Xs = X
                scaler_transform_failed_any = True
        else:
            Xs = X

        Xs, used_em = align_X_to_n_features(Xs, n_features_model, allow=allow_align)
        if used_em:
            emergency_align_used = True

        vol = (
            float(last["ewma_vol_logret_14"])
            if "ewma_vol_logret_14" in last.index and pd.notna(last["ewma_vol_logret_14"])
            else 0.02
        )
        vol = max(vol, 0.005)
        if step == 0:
            ewma_vol_start = vol

        daily_cap = daily_log_return_cap(vol, historical_abs_logret_p99=float(p99_hist) if p99_hist is not None else None)

        raw_preds: dict[str, float] = {}
        preds_lr: dict[str, float] = {}
        for name, model in models.items():
            try:
                X_df = prediction_matrix_to_dataframe(
                    Xs,
                    model_wrapper=model,
                    training_feature_columns=columns,
                )
                r_raw = float(np.ravel(model.predict(X_df))[0])
                raw_preds[name] = r_raw
                r = clip_daily_log_return(r_raw, daily_cap, outer_cap=OUTER_DAILY_ABS_CAP)
                preds_lr[name] = r
            except Exception as e:
                logger.warning("Model %s predict failed at step %d: %s", name, step, e)

        if step == 0 and raw_preds:
            logger.debug(
                "Recursive forecast step0 raw log-returns by model: %s (daily_cap=%.5f, daily_vol=%.5f)",
                {k: round(v, 6) for k, v in raw_preds.items()},
                daily_cap,
                vol,
            )

        if preds_lr and all(abs(float(r)) >= daily_cap - 1e-6 for r in preds_lr.values()):
            saturation_steps += 1

        if not preds_lr:
            break

        for name in models:
            if name in raw_preds:
                model_raw_by_step[name].append(raw_preds[name])

        r_w = float(
            np.clip(
                _weighted_log_return(preds_lr, weights),
                -OUTER_DAILY_ABS_CAP,
                OUTER_DAILY_ABS_CAP,
            )
        )
        r_w = clip_daily_log_return(r_w, daily_cap, outer_cap=OUTER_DAILY_ABS_CAP)

        last_close = float(history["close"].iloc[-1])
        next_price = last_close * math.exp(r_w)

        band_scale = 1.0
        if "vol_regime_encoded" in last.index and pd.notna(last["vol_regime_encoded"]):
            vr = float(last["vol_regime_encoded"])
            # Tighter bands in calm regimes, wider in high vol (more informative uncertainty).
            band_scale = 0.88 + 0.12 * max(0.0, min(2.0, vr))
        hw_log = vol_k * vol * math.sqrt(float(step + 1)) * band_scale
        lower = next_price * math.exp(-hw_log)
        upper = next_price * math.exp(hw_log)

        for n in list(model_prices.keys()):
            if n in preds_lr:
                model_prices[n] = float(model_prices[n] * math.exp(preds_lr[n]))

        last_d = pd.to_datetime(history["date"].iloc[-1])
        next_d = last_d + pd.Timedelta(days=1)
        last_vol = float(history["volume"].iloc[-1])
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "date": next_d,
                            "open": next_price,
                            "high": next_price,
                            "low": next_price,
                            "close": next_price,
                            "volume": last_vol,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        agree = model_output_diversity_agreement(
            list(preds_lr.values()),
            -OUTER_DAILY_ABS_CAP,
            OUTER_DAILY_ABS_CAP,
        )

        path.append({
            "day_index": step + 1,
            "predicted_log_return": round(r_w, 8),
            "predicted_simple_return": round(math.exp(r_w) - 1.0, 8),
            "predicted_price": round(next_price, 8),
            "lower_bound": round(lower, 8),
            "upper_bound": round(upper, 8),
            "model_log_returns": {k: round(v, 8) for k, v in preds_lr.items()},
            "model_prices": {k: round(model_prices[k], 8) for k in preds_lr},
            "ensemble_prediction": round(next_price, 8),
            "agreement_score": round(agree, 4),
            "ewma_vol_logret": round(vol, 8),
        })

    guardrail_cum = apply_cumulative_guardrail_to_path(
        path,
        spot_start,
        ewma_vol_start,
        meta,
        vol_k=vol_k,
    )

    out_signals = normalize_forecast_path_signals(
        compute_path_output_signals(
            path,
            model_raw_by_step,
            max_missing_feature_fill_ratio=max_miss_fill,
            emergency_width_align_used=emergency_align_used,
            spot_start=spot_start,
        )
    )

    has_saved = bool(saved_cols)
    row_ix = feat_df0.iloc[-1].index if len(feat_df0) else feat_df0.columns
    exact_cols = exact_training_columns_present(saved_cols, row_ix) if saved_cols else False

    quality = classify_forecast_quality(
        has_scaler=scaler is not None,
        has_saved_feature_columns=has_saved,
        emergency_width_align_used=emergency_align_used,
        scaler_transform_failed_any=scaler_transform_failed_any,
    )

    sanity = summarize_path_sanity(path, spot_start, meta)

    diagnostics = _normalize_recursive_forecast_diagnostics(
        {
            "forecast_quality": quality,
            "used_scaler": scaler is not None,
            "exact_feature_match": exact_cols and not emergency_align_used,
            "fallback_mode": quality != "production",
            "used_emergency_width_align": emergency_align_used,
            "scaler_transform_failed": scaler_transform_failed_any,
            "realism_guardrail_cumulative": guardrail_cum,
            "target_log_return_1d": meta.get("training_target", "target_log_return_1d"),
            "clip_saturation_step_fraction": round(saturation_steps / max(len(path), 1), 4),
            "training_metadata_present": bool(meta),
            "sanity_check": sanity,
            "sanity_extreme": sanity.get("status") == "extreme_outlier",
            "is_constant_prediction": out_signals["is_constant_prediction"],
            "low_variance_warning": out_signals["low_variance_warning"],
            "degraded_input": out_signals["degraded_input"],
            "ensemble_log_return_var": out_signals["ensemble_log_return_var"],
            "price_relative_variance": out_signals["price_relative_variance"],
            "model_horizon_variance_raw": out_signals["model_horizon_variance_raw"],
            "step0_raw_logret_by_model": out_signals["step0_raw_logret_by_model"],
            "max_missing_feature_fill_ratio": out_signals["max_missing_feature_fill_ratio"],
        }
    )

    return path, model_prices, ewma_vol_start, diagnostics
