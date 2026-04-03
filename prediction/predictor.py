"""
Prediction service: log-return models with recursive multi-step price forecasting.

Models predict next-day log return; horizon is built by chaining price updates and
recomputing features. No synthetic path smoothing or output inertia.
"""

from __future__ import annotations

import math
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from data.data_refresh import get_last_refresh_time, refresh_data_if_needed
from features.feature_builder import build_features, get_feature_columns
from models.model_registry import get_model, list_models
from prediction.ensemble_weights import compute_weights_from_metrics, load_evaluation_metrics
from prediction.forecast_intel import (
    apply_confidence_penalty_caps,
    build_multi_horizon,
    build_prediction_explanation,
    classify_trend_from_forecast_path,
    classify_volatility_level,
    compute_financial_confidence,
    compute_high_conviction,
    compute_signal_strength_score,
    finalize_forecast_confidence,
    infer_volatility_regime_from_context,
)
from prediction.forecast_postprocess import market_data_freshness
from prediction.inference_utils import (
    align_X_to_n_features,
    feature_row_to_matrix,
    infer_n_features_for_ensemble,
    load_scaler_bundle_from_disk,
    load_training_metadata,
    prediction_matrix_to_dataframe,
    resolve_inference_columns,
)
from prediction.artifact_audit import (
    evaluate_artifact_bundle,
    forecast_audit_api_subset,
    resolve_forecast_artifact_mode,
)
from prediction.forecast_validity import (
    compute_forecast_validity_and_quality_score,
    confidence_composition_doc,
)
from prediction.price_format import round_price as round_price_for_display
from prediction.recursive_forecast import DEFAULT_HORIZON, recursive_log_return_forecast
from utils.config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

HORIZON_DAYS = DEFAULT_HORIZON


def _model_agreement_score(predictions: list[float]) -> float:
    if not predictions:
        return 0.0
    arr = np.array(predictions)
    mean = arr.mean()
    if abs(mean) < 1e-10:
        return 1.0 if arr.std() < 1e-10 else 0.0
    cv = arr.std() / abs(mean)
    return max(0.0, min(1.0, 1.0 - cv))


def _weighted_average(values_by_model: dict[str, float], weights: dict[str, float]) -> float:
    if not values_by_model:
        return 0.0
    wsum = 0.0
    xsum = 0.0
    for name, v in values_by_model.items():
        w = float(weights.get(name, 0.0))
        xsum += w * float(v)
        wsum += w
    if wsum < 1e-12:
        return float(np.mean(list(values_by_model.values())))
    return float(xsum / wsum)


class Predictor:
    """
    Loads per-coin scalers and log-return models; builds forecasts via recursive_forecast.
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        evaluation_dir: Path | None = None,
    ):
        self.models_dir = Path(models_dir) if models_dir else settings.training.models_dir
        self.evaluation_dir = Path(evaluation_dir) if evaluation_dir else (
            settings.training.features_dir.parent / "evaluation"
        )
        self._loaded: dict[str, dict[str, Any]] = {}
        self._last_forecast_error: dict[str, Any] | None = None

    def get_last_forecast_error(self) -> dict[str, Any] | None:
        """Structured failure from the last predict_from_latest_features (if any)."""
        return self._last_forecast_error

    def load_models(self, coin: str) -> bool:
        coin_dir = self.models_dir / coin.replace(" ", "_")
        if not coin_dir.exists():
            logger.warning("No model directory for %s at %s", coin, coin_dir)
            return False
        self._loaded[coin] = {}
        for name in list_models():
            path = coin_dir / f"{name}.joblib"
            if path.exists():
                try:
                    model = get_model(name)
                    model.load(path)
                    self._loaded[coin][name] = model
                    inner = getattr(model, "model", model)
                    logger.debug(
                        "Loaded %s for %s from %s (estimator_type=%s id=%s)",
                        name,
                        coin,
                        path,
                        type(inner).__name__,
                        id(inner),
                    )
                except Exception as e:
                    logger.exception("Failed to load %s for %s: %s", name, coin, e)
        dc_path = coin_dir / "directional_classifier.joblib"
        if dc_path.exists():
            try:
                self._loaded[coin]["directional_classifier"] = joblib.load(dc_path)
                logger.info("Loaded directional_classifier for %s", coin)
            except Exception as e:
                logger.warning("Failed to load directional_classifier for %s: %s", coin, e)
        return any(name in self._loaded[coin] for name in list_models())

    def _directional_classifier_proba(
        self,
        feature_df: pd.DataFrame,
        coin: str,
        bundle: dict[str, Any],
    ) -> dict[str, float] | None:
        """Posterior over down/neutral/up from optional logistic head (scaled features)."""
        clf = (self._loaded.get(coin) or {}).get("directional_classifier")
        cols = bundle.get("feature_columns")
        scaler = bundle.get("scaler")
        if clf is None or not cols:
            return None
        row = feature_df.iloc[-1]
        try:
            vals = []
            for c in cols:
                if c not in feature_df.columns:
                    return None
                v = row[c]
                vals.append(float(v) if pd.notna(v) else 0.0)
            X = np.asarray(vals, dtype=np.float64).reshape(1, -1)
        except (TypeError, ValueError, KeyError):
            return None
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception as e:
                logger.warning("directional_classifier scaler transform failed: %s", e)
                return None
        try:
            proba = clf.predict_proba(X)[0]
            classes = list(getattr(clf, "classes_", range(len(proba))))
            labels = ("down", "neutral", "up")
            out: dict[str, float] = {}
            for i, cls in enumerate(classes):
                idx = int(cls)
                if 0 <= idx < 3:
                    out[labels[idx]] = float(proba[i])
            return out if out else None
        except Exception as e:
            logger.warning("directional_classifier predict_proba failed: %s", e)
            return None

    def _load_inference_bundle(self, coin: str) -> dict[str, Any]:
        """
        Load scaler + saved feature column order when artifacts exist.
        If nothing on disk, returns scaler=None (raw-feature inference for legacy models).
        """
        slug = coin.replace(" ", "_")
        coin_dir = self.models_dir / slug
        bundle = load_scaler_bundle_from_disk(coin_dir, slug)
        if bundle.get("source_path"):
            logger.info("Loaded inference bundle for %s from %s", coin, bundle["source_path"])
        else:
            logger.warning(
                "[legacy-inference] No scaler artifact for %s under %s — "
                "forecast continues using raw aligned features (not a hard failure). "
                "Retrain to save scaler: python scripts/train_pipeline.py --coin %s",
                coin,
                coin_dir,
                coin,
            )
        return bundle

    def predict(
        self,
        coin: str,
        X: pd.DataFrame | np.ndarray,
        current_close: float | None = None,
        feature_context: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Single-step ensemble log return from one scaled feature row; maps to price if spot given.
        """
        if coin not in self._loaded:
            self.load_models(coin)
        models = self._loaded.get(coin, {})
        bundle = self._load_inference_bundle(coin)
        if not models:
            return {
                "coin": coin,
                "predictions": {},
                "average_prediction": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "model_agreement_score": 0.0,
                "horizon_days": HORIZON_DAYS,
                "_path": None,
                "_path_capable": False,
            }

        scaler = bundle.get("scaler")
        if isinstance(X, pd.DataFrame):
            row = X.iloc[-1]
            cols = resolve_inference_columns(bundle, row.index)
            Xa = feature_row_to_matrix(row, cols)
        else:
            saved = bundle.get("feature_columns")
            cols = list(saved) if saved else get_feature_columns(include_targets=False)
            Xa = np.asarray(X, dtype=np.float64).reshape(1, -1)
        Xa = np.nan_to_num(Xa, nan=0.0, posinf=0.0, neginf=0.0)
        if scaler is not None:
            try:
                Xs = scaler.transform(Xa)
            except Exception as e:
                logger.warning("Scaler transform failed in predict(); using raw features: %s", e)
                Xs = Xa
        else:
            Xs = Xa

        Xs, _ = align_X_to_n_features(
            Xs,
            infer_n_features_for_ensemble(models),
            allow=settings.prediction.allow_emergency_feature_align,
        )

        metrics = load_evaluation_metrics(self.evaluation_dir, coin)
        weights = compute_weights_from_metrics(list(models.keys()), metrics)
        preds_lr: dict[str, float] = {}
        for name, model in models.items():
            try:
                X_df = prediction_matrix_to_dataframe(
                    Xs,
                    model_wrapper=model,
                    training_feature_columns=cols,
                )
                r = float(np.ravel(model.predict(X_df))[0])
                r = float(np.clip(r, -0.15, 0.15))
                preds_lr[name] = r
            except Exception as e:
                logger.warning("Model %s predict failed: %s", name, e)
        if not preds_lr:
            return {
                "coin": coin,
                "predictions": {},
                "average_prediction": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "model_agreement_score": 0.0,
                "horizon_days": HORIZON_DAYS,
                "_path": None,
                "_path_capable": False,
            }

        r_w = float(np.clip(_weighted_average(preds_lr, weights), -0.15, 0.15))
        spot = float(current_close) if current_close and current_close > 0 else 1.0
        p_ens = spot * float(np.exp(r_w))
        vol = float((feature_context or {}).get("ewma_vol_logret_14", 0.02) or 0.02)
        hw = 2.0 * max(vol, 0.005)
        lo, hi = p_ens * np.exp(-hw), p_ens * np.exp(hw)
        mp_prices = {n: spot * float(np.exp(r)) for n, r in preds_lr.items()}

        return {
            "coin": coin,
            "predictions": mp_prices,
            "average_prediction": p_ens,
            "lower_bound": float(lo),
            "upper_bound": float(hi),
            "model_agreement_score": _model_agreement_score(list(preds_lr.values())),
            "horizon_days": HORIZON_DAYS,
            "_path": None,
            "_path_capable": False,
            "_model_weights": weights,
            "_predicted_log_return": r_w,
        }

    def _feature_context_from_row(self, df: pd.DataFrame) -> dict[str, float]:
        keys = (
            "atr_14",
            "rolling_volatility_14",
            "ewma_vol_logret_14",
            "ewma_vol_logret_30",
            "rolling_std_logret_14",
            "vol_regime_encoded",
            "rsi_14",
            "return_3d",
            "return_7d",
            "return_14d",
            "daily_return",
            "macd_histogram",
            "adx_14",
            "trend_strength_adx",
            "momentum_logret_7d",
            "close",
        )
        out: dict[str, float] = {}
        for k in keys:
            if k in df.columns:
                try:
                    v = df[k].iloc[-1]
                    if pd.notna(v):
                        out[k] = float(v)
                except (TypeError, ValueError):
                    continue
        return out

    def predict_from_latest_features(
        self,
        coin: str,
        features_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Recursive 14-step forecast from latest feature parquet OHLCV history.
        """
        self._last_forecast_error = None
        features_dir = Path(features_dir) if features_dir else settings.training.features_dir
        refresh_data_if_needed(coin, features_dir=features_dir)
        path = features_dir / f"{coin.replace(' ', '_')}_features.parquet"
        if not path.exists():
            logger.warning("Feature file not found: %s", path)
            self._last_forecast_error = {
                "code": "no_feature_file",
                "message": f"Feature parquet not found: {path.name}",
                "hint": "Run training pipeline or POST /api/refresh/{coin} to build features.",
            }
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.exception("Failed to load features for %s: %s", coin, e)
            self._last_forecast_error = {
                "code": "feature_load_failed",
                "message": str(e),
                "hint": "Check that the parquet file is readable and not corrupted.",
            }
            return None
        if df.empty or len(df) < 80:
            logger.warning("Insufficient history for %s", coin)
            self._last_forecast_error = {
                "code": "insufficient_history",
                "message": f"Need at least 80 rows of history; got {len(df)}.",
                "hint": "Fetch more historical data and rebuild features.",
            }
            return None
        req = ["date", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in req):
            logger.warning("Feature file missing OHLCV columns for %s", coin)
            self._last_forecast_error = {
                "code": "missing_ohlcv_columns",
                "message": f"Parquet must include columns: {req}",
                "hint": "Regenerate features from OHLCV source data.",
            }
            return None

        if coin not in self._loaded:
            self.load_models(coin)
        models = self._loaded.get(coin, {})
        bundle = self._load_inference_bundle(coin)
        if not models:
            self._last_forecast_error = {
                "code": "no_models",
                "message": f"No trained model files in {self.models_dir / coin.replace(' ', '_')}.",
                "hint": "Run: python scripts/train_pipeline.py --coin " + coin,
            }
            return None

        metrics = load_evaluation_metrics(self.evaluation_dir, coin)
        weights = compute_weights_from_metrics(list(models.keys()), metrics)

        slug = coin.replace(" ", "_")
        coin_dir = self.models_dir / slug
        training_meta = load_training_metadata(coin_dir)

        history = df[req].copy()
        feat_live = build_features(history.tail(min(1000, len(history))), include_targets=False)
        live_ix = feat_live.iloc[-1].index if len(feat_live) else pd.Index([])
        bundle_audit = evaluate_artifact_bundle(
            coin_dir,
            slug,
            self.evaluation_dir,
            live_feature_columns=live_ix,
        )

        inf_bundle = self._load_inference_bundle(coin)
        dir_probs = self._directional_classifier_proba(df, coin, inf_bundle)

        try:
            path_raw, _, vol0, forecast_diag = recursive_log_return_forecast(
                history,
                models,
                weights,
                bundle,
                training_meta=training_meta,
                horizon=HORIZON_DAYS,
            )
        except Exception as e:
            logger.exception("Recursive forecast failed for %s: %s", coin, e)
            self._last_forecast_error = {
                "code": "recursive_forecast_failed",
                "message": str(e),
                "hint": "Check feature schema matches training and models are compatible.",
            }
            return None

        if len(path_raw) != HORIZON_DAYS:
            logger.warning("Incomplete path for %s (%d days)", coin, len(path_raw))
            self._last_forecast_error = {
                "code": "incomplete_path",
                "message": f"Forecast stopped early with {len(path_raw)} of {HORIZON_DAYS} days (model or feature failure).",
                "hint": "Inspect logs for predict failures; verify models accept current feature dimension.",
            }
            return None

        current_close = float(df["close"].iloc[-1])
        feature_context = self._feature_context_from_row(df)
        feature_context.setdefault("ewma_vol_logret_14", vol0)

        last = path_raw[-1]
        agreements = [float(p["agreement_score"]) for p in path_raw]
        mean_agree = float(np.mean(agreements)) if agreements else 0.0
        lr_matrix = [
            [float(p["model_log_returns"][m]) for m in sorted(p["model_log_returns"])]
            for p in path_raw
        ]
        implied_ret = (
            (float(last["predicted_price"]) - current_close) / current_close
            if current_close > 0
            else 0.0
        )
        implied_lr = math.log(
            max(float(last["predicted_price"]), 1e-30) / max(current_close, 1e-30)
        )
        sig_strength = compute_signal_strength_score(
            current_close, float(last["predicted_price"]), feature_context
        )
        vr_enc = feature_context.get("vol_regime_encoded")

        merged = dict(forecast_diag)
        merged["forecast_audit"] = bundle_audit
        merged["artifact_mode"] = resolve_forecast_artifact_mode(bundle_audit, forecast_diag)
        merged["schema_match"] = bool(
            bundle_audit.get("schema_version_match")
            and bundle_audit.get("columns_match_official_list")
            and bundle_audit.get("live_columns_cover_saved_features")
        )
        merged["artifact_bundle_tier"] = bundle_audit.get("artifact_bundle_tier", "legacy")
        merged["feature_schema_version_artifact"] = bundle_audit.get("feature_schema_version_artifact")
        merged["feature_schema_version_expected"] = bundle_audit.get("feature_schema_version_expected")
        _fq = {"production": "production", "degraded": "degraded", "legacy": "legacy-fallback"}
        merged["forecast_quality"] = _fq.get(str(merged["artifact_mode"]), "degraded")
        merged["fallback_mode"] = merged["artifact_mode"] != "production"
        merged["directional_probabilities"] = dir_probs or {}
        merged["signal_strength_score"] = round(max(0.0, float(sig_strength)), 4)
        merged["volatility_regime"] = infer_volatility_regime_from_context(feature_context)
        log_v_hc = float(
            feature_context.get("ewma_vol_logret_14")
            or feature_context.get("rolling_std_logret_14")
            or feature_context.get("ewma_vol_logret_30")
            or 0.018
        )
        trend_for_hc, _ = classify_trend_from_forecast_path(
            current_close, path_raw, log_vol_daily=log_v_hc
        )
        merged["high_conviction"] = bool(
            compute_high_conviction(
                signal_strength=sig_strength,
                mean_path_agreement=mean_agree,
                feature_context=feature_context,
                implied_horizon_return=implied_ret,
                directional_probs=dir_probs,
                trend_label=trend_for_hc,
            )
        )
        validity, qscore = compute_forecast_validity_and_quality_score(merged)
        merged["forecast_validity"] = validity
        merged["forecast_quality_score"] = qscore
        merged["confidence_composition_reference"] = confidence_composition_doc()

        fq_conf = str(merged.get("forecast_quality", "degraded"))
        raw_conf = compute_financial_confidence(
            agreements,
            lr_matrix,
            metrics,
            float(vol0),
            forecast_quality=fq_conf,
            fallback_mode=bool(merged.get("fallback_mode", True)),
            artifact_mode=str(merged.get("artifact_mode", "legacy")),
            guardrail_cumulative=bool(merged.get("realism_guardrail_cumulative")),
            clip_saturation_fraction=float(merged.get("clip_saturation_step_fraction", 0.0)),
            exact_feature_match=bool(merged.get("exact_feature_match", False)),
            signal_strength=sig_strength,
            directional_probs=dir_probs,
            implied_horizon_log_return=implied_lr,
            vol_regime_encoded=float(vr_enc) if vr_enc is not None else None,
        )
        merged["confidence_score"] = finalize_forecast_confidence(
            apply_confidence_penalty_caps(raw_conf, merged),
            mean_path_agreement=mean_agree,
            directional_probs=dir_probs,
            vol_regime=str(merged.get("volatility_regime") or "MEDIUM"),
        )

        return {
            "coin": coin,
            "predictions": last["model_prices"],
            "average_prediction": float(last["predicted_price"]),
            "lower_bound": float(last["lower_bound"]),
            "upper_bound": float(last["upper_bound"]),
            "model_agreement_score": float(last["agreement_score"]),
            "horizon_days": HORIZON_DAYS,
            "_path": path_raw,
            "_path_capable": True,
            "_model_weights": weights,
            "_current_close": current_close,
            "_last_date": df["date"].iloc[-1] if "date" in df.columns else None,
            "_feature_context": feature_context,
            "_forecast_vol_start": vol0,
            "_mean_path_agreement": mean_agree,
            "_forecast_diagnostics": merged,
        }

    def forecast_path(
        self,
        coin: str,
        features_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        raw = self.predict_from_latest_features(coin, features_dir=features_dir)
        if raw is None:
            return None
        path_data = raw.get("_path")
        if not path_data or len(path_data) != HORIZON_DAYS:
            self._last_forecast_error = {
                "code": "invalid_forecast_payload",
                "message": f"Expected {HORIZON_DAYS}-day path; got {len(path_data or [])}.",
                "hint": "See server logs for recursive forecast or API errors.",
            }
            return None

        weights = raw.get("_model_weights") or {}
        today_utc = datetime.now(timezone.utc).date()
        window_start = today_utc + timedelta(days=1)

        latest_market_ts: str | None = None
        last_date = raw.get("_last_date")
        if last_date is not None:
            try:
                ts = pd.to_datetime(last_date)
                latest_market_ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            except Exception:
                latest_market_ts = str(last_date) if last_date else None

        data_freshness, data_age_hours, data_freshness_extra = market_data_freshness(latest_market_ts)

        current = float(raw.get("_current_close") or 0)
        price_ref = current if current > 0 else 1.0

        forecast_path: list[dict[str, Any]] = []
        for d, p in enumerate(path_data):
            fc_date = window_start + timedelta(days=d)
            fc_dt = datetime.combine(fc_date, time.min, tzinfo=timezone.utc)
            fc_ts = fc_dt.isoformat()
            forecast_path.append({
                "day_index": d + 1,
                "forecast_date": fc_date.isoformat(),
                "forecast_timestamp": fc_ts,
                "predicted_price": round_price_for_display(float(p["predicted_price"]), price_ref),
                "predicted_log_return": round(float(p["predicted_log_return"]), 8),
                "predicted_return": round(float(p["predicted_simple_return"]), 8),
                "lower_bound": round_price_for_display(float(p["lower_bound"]), price_ref),
                "upper_bound": round_price_for_display(float(p["upper_bound"]), price_ref),
                "model_predictions": {
                    k: round_price_for_display(float(v), price_ref) for k, v in p["model_prices"].items()
                },
                "ensemble_prediction": round_price_for_display(float(p["ensemble_prediction"]), price_ref),
                "agreement_score": round(float(p["agreement_score"]), 4),
                "volatility": round(float(p["ewma_vol_logret"]), 8),
            })

        prices = [x["predicted_price"] for x in forecast_path]
        min_p, max_p = min(prices), max(prices)
        final = forecast_path[-1]["predicted_price"]
        trend = "up" if final > current else "down" if final < current else "flat" if current > 0 else "flat"

        fc_ctx = raw.get("_feature_context") or {}
        metrics = load_evaluation_metrics(self.evaluation_dir, coin)
        rv = fc_ctx.get("rolling_volatility_14")
        log_v_trend = float(
            fc_ctx.get("ewma_vol_logret_14")
            or fc_ctx.get("rolling_std_logret_14")
            or fc_ctx.get("ewma_vol_logret_30")
            or 0.018
        )
        trend_label, _ = classify_trend_from_forecast_path(
            float(current),
            path_data,
            log_vol_daily=log_v_trend,
        )

        vol_start = float(raw.get("_forecast_vol_start") or fc_ctx.get("ewma_vol_logret_14") or 0.02)
        agreements = [float(x["agreement_score"]) for x in forecast_path]
        diag = raw.get("_forecast_diagnostics") or {}
        fq = str(diag.get("forecast_quality", "degraded"))
        conf_pre = diag.get("confidence_score")
        if conf_pre is not None:
            confidence = float(conf_pre)
        else:
            lr_matrix_fb = [
                [float(p["model_log_returns"][m]) for m in sorted(p["model_log_returns"])]
                for p in path_data
            ]
            vr_raw = fc_ctx.get("vol_regime_encoded")
            confidence = compute_financial_confidence(
                agreements,
                lr_matrix_fb,
                metrics,
                vol_start,
                forecast_quality=fq,
                fallback_mode=bool(diag.get("fallback_mode", True)),
                artifact_mode=str(diag.get("artifact_mode", "legacy")),
                guardrail_cumulative=bool(diag.get("realism_guardrail_cumulative")),
                clip_saturation_fraction=float(diag.get("clip_saturation_step_fraction", 0.0)),
                exact_feature_match=bool(diag.get("exact_feature_match", False)),
                signal_strength=float(diag.get("signal_strength_score", 0.0)),
                directional_probs=diag.get("directional_probabilities"),
                implied_horizon_log_return=math.log(
                    max(float(final), 1e-30) / max(current, 1e-30)
                )
                if current > 0
                else 0.0,
                vol_regime_encoded=float(vr_raw) if vr_raw is not None else None,
            )
            confidence = apply_confidence_penalty_caps(confidence, diag)
            confidence = finalize_forecast_confidence(
                confidence,
                mean_path_agreement=float(raw.get("_mean_path_agreement") or np.mean(agreements)),
                directional_probs=diag.get("directional_probabilities"),
                vol_regime=str(diag.get("volatility_regime") or infer_volatility_regime_from_context(fc_ctx)),
            )

        volatility_level = classify_volatility_level(rv, current if current else None)
        if volatility_level == "UNKNOWN":
            volatility_level = infer_volatility_regime_from_context(fc_ctx)
        mean_agree = float(raw.get("_mean_path_agreement") or np.mean(agreements))
        ilr_explain = (
            math.log(max(float(path_data[-1]["predicted_price"]), 1e-30) / max(float(current), 1e-30))
            if current > 0
            else 0.0
        )
        explanation = build_prediction_explanation(
            fc_ctx,
            trend_label,
            confidence,
            mean_agree,
            weights,
            directional_probs=diag.get("directional_probabilities"),
            implied_horizon_log_return=ilr_explain,
        )
        if bool(diag.get("is_constant_prediction")):
            explanation += (
                " The ensemble produced an effectively flat path (constant or near-constant "
                "log-returns); treat this forecast as invalid for trading decisions until models are retrained."
            )
        elif bool(diag.get("low_variance_warning")):
            explanation += " Forecast path shows unusually low variance; uncertainty is elevated."
        if bool(diag.get("degraded_input")):
            explanation += " Many features were missing or width-aligned (zero-filled); retrain with current feature schema."

        multi_horizon = build_multi_horizon(forecast_path, current)

        msg_parts = [data_freshness_extra] if data_freshness_extra else []
        if data_freshness == "stale":
            msg_parts.append("Prediction is based on features that may not include the latest session.")
        combined_msg = " ".join(m for m in msg_parts if m) or None

        forecast_audit_flat = forecast_audit_api_subset(diag.get("forecast_audit") or {})

        return {
            "coin": coin,
            "current_price": round_price_for_display(current, price_ref),
            "horizon_days": HORIZON_DAYS,
            "model_agreement_score": round(float(forecast_path[-1]["agreement_score"]), 4),
            "latest_market_timestamp": latest_market_ts,
            "forecast_period_start": forecast_path[0]["forecast_date"],
            "forecast_period_end": forecast_path[-1]["forecast_date"],
            "forecast_path": forecast_path,
            "summary": {
                "final_day_prediction": round_price_for_display(float(final), price_ref),
                "min_forecast_price": round_price_for_display(float(min_p), price_ref),
                "max_forecast_price": round_price_for_display(float(max_p), price_ref),
                "average_forecast_price": round_price_for_display(float(np.mean(prices)), price_ref),
                "trend_direction_14d": trend,
                "trend_label": trend_label,
            },
            "confidence_score": round(confidence, 4),
            "trend_label": trend_label,
            "explanation": explanation,
            "model_weights": {k: round(float(v), 6) for k, v in sorted(weights.items())},
            "volatility_level": volatility_level,
            "multi_horizon": multi_horizon,
            "mean_path_agreement": round(mean_agree, 4),
            "forecast_volatility": round(vol_start, 8),
            "data_freshness": data_freshness,
            "data_age_hours": data_age_hours,
            "data_freshness_detail": combined_msg,
            "last_refresh_time": get_last_refresh_time(coin),
            "forecast_quality": fq,
            "artifact_mode": str(diag.get("artifact_mode", "legacy")),
            "schema_match": bool(diag.get("schema_match", False)),
            "scaler_used": bool(diag.get("used_scaler", False)),
            "used_scaler": bool(diag.get("used_scaler", False)),
            "exact_feature_match": bool(diag.get("exact_feature_match", False)),
            "fallback_mode": bool(diag.get("fallback_mode", True)),
            "forecast_validity": str(diag.get("forecast_validity", "questionable")),
            "forecast_quality_score": float(diag.get("forecast_quality_score") or 0.0),
            "confidence_composition_reference": str(diag.get("confidence_composition_reference") or ""),
            "realism_guardrail_triggered": bool(diag.get("realism_guardrail_cumulative")),
            "forecast_audit": forecast_audit_flat,
            "forecast_diagnostics": {
                "used_emergency_width_align": bool(diag.get("used_emergency_width_align")),
                "scaler_transform_failed": bool(diag.get("scaler_transform_failed")),
                "clip_saturation_step_fraction": float(diag.get("clip_saturation_step_fraction", 0.0)),
                "training_metadata_present": bool(diag.get("training_metadata_present")),
                "sanity_check": diag.get("sanity_check") or {},
                "step0_raw_logret_by_model": diag.get("step0_raw_logret_by_model") or {},
                "model_horizon_variance_raw": diag.get("model_horizon_variance_raw") or {},
                "ensemble_log_return_var": float(diag.get("ensemble_log_return_var", 0.0)),
                "price_relative_variance": float(diag.get("price_relative_variance", 0.0)),
                "max_missing_feature_fill_ratio": float(diag.get("max_missing_feature_fill_ratio", 0.0)),
            },
            "is_constant_prediction": bool(diag.get("is_constant_prediction", False)),
            "low_variance_warning": bool(diag.get("low_variance_warning", False)),
            "degraded_input": bool(diag.get("degraded_input", False)),
            "high_conviction": bool(diag.get("high_conviction", False)),
            "signal_strength_score": float(diag.get("signal_strength_score") or 0.0),
            "directional_probabilities": dict(diag.get("directional_probabilities") or {}),
            "volatility_regime": str(
                diag.get("volatility_regime") or infer_volatility_regime_from_context(fc_ctx)
            ),
        }
