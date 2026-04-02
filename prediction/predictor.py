"""
Prediction service: loads trained models and produces structured forecasts.

Supports weighted ensemble (from evaluation metrics), multi-output 14-day path,
volatility-aware bounds, and metadata for API / dashboard intelligence layer.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.feature_builder import get_feature_columns
from models.model_registry import get_model, list_models
from prediction.ensemble_weights import compute_weights_from_metrics, load_evaluation_metrics
from prediction.forecast_intel import (
    build_multi_horizon,
    build_prediction_explanation,
    classify_trend_label,
    classify_volatility_level,
    compute_confidence_score,
)
from prediction.forecast_postprocess import (
    apply_sanity_pass,
    market_data_freshness,
    mean_path_model_relative_spread,
    recenter_and_widen_bounds,
    scale_model_predictions,
    smooth_path_from_anchor,
)
from prediction.prediction_state import (
    blend_path_with_previous,
    load_last_path,
    load_last_trend_score,
    save_state,
)
from data.data_refresh import get_last_refresh_time, refresh_data_if_needed
from utils.config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

HORIZON_DAYS = 14

# Bounds: blend model spread with ATR- and volatility-scaled bands
_ATR_K = 1.5
_VOL_K = 1.25
_MIN_RANGE_FRAC = 0.005  # at least 0.5% of spot


def _model_agreement_score(predictions: list[float]) -> float:
    """0–1 score: higher when models agree (lower std relative to mean)."""
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
        vals = list(values_by_model.values())
        return float(np.mean(vals))
    return float(xsum / wsum)


def _volatility_aware_bounds(
    weighted_mean: float,
    model_preds: dict[str, float],
    day_index_1based: int,
    close: float | None,
    feature_context: dict[str, float] | None,
) -> tuple[float, float]:
    vals = list(model_preds.values())
    if len(vals) >= 2:
        spread = max(vals) - min(vals)
    else:
        spread = abs(weighted_mean) * 0.01

    c = float(close) if close is not None and close > 0 else max(abs(weighted_mean), 1.0)
    fc = feature_context or {}
    atr = float(fc.get("atr_14", 0) or 0)
    roll_vol = float(fc.get("rolling_volatility_14", 0.02) or 0.02)
    day_scale = float(day_index_1based) ** 0.5

    atr_band = _ATR_K * atr * day_scale if atr > 0 else 0.0
    vol_band = c * max(roll_vol, 1e-6) * day_scale * _VOL_K
    min_frac = c * _MIN_RANGE_FRAC
    half_width = max(spread / 2.0, atr_band, vol_band, min_frac)
    return weighted_mean - half_width, weighted_mean + half_width


class Predictor:
    """
    Loads trained models per coin and runs ensemble predictions.
    Supports multi-output (14-day path) with metric-weighted ensemble and
    volatility-aware prediction intervals.
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
        self._forecast_state_dir = settings.training.features_dir.parent / "cache" / "forecast_state"
        self._loaded: dict[str, dict[str, Any]] = {}

    def load_models(self, coin: str) -> bool:
        """Load models for a coin. Returns True if any model loaded."""
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
                except Exception as e:
                    logger.exception("Failed to load %s for %s: %s", name, coin, e)
        return len(self._loaded.get(coin, {})) > 0

    def predict(
        self,
        coin: str,
        X: pd.DataFrame | np.ndarray,
        current_close: float | None = None,
        feature_context: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Run ensemble prediction for a coin.
        Returns multi-output (14-day path) if models support it, else single-point.
        """
        if coin not in self._loaded:
            self.load_models(coin)
        models = self._loaded.get(coin, {})
        if not models:
            return {
                "coin": coin,
                "predictions": {},
                "average_prediction": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "model_agreement_score": 0.0,
                "horizon_days": HORIZON_DAYS,
            }

        model_outputs: dict[str, np.ndarray] = {}
        for name, model in models.items():
            try:
                vals = model.predict(X)
                arr = np.atleast_1d(vals)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1) if arr.size > 1 else arr.reshape(1, 1)
                model_outputs[name] = arr
            except Exception as e:
                logger.warning("Model %s predict failed for %s: %s", name, coin, e)

        if not model_outputs:
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

        multi_out: dict[str, np.ndarray] = {}
        single_out: dict[str, np.ndarray] = {}
        for name, arr in model_outputs.items():
            try:
                n_out = int(arr.shape[-1]) if arr.size > 0 else 0
                shape_str = str(tuple(arr.shape))
            except (IndexError, TypeError):
                n_out = 0
                shape_str = "unknown"
            if n_out >= HORIZON_DAYS:
                multi_out[name] = arr
                logger.info(
                    "Model %s for %s: output_shape=%s, multi-output (n=%d) -> accepted for forecast path",
                    name, coin, shape_str, n_out,
                )
            else:
                single_out[name] = arr
                logger.info(
                    "Model %s for %s: output_shape=%s, single-output (n=%d) -> skipped for forecast path",
                    name, coin, shape_str, n_out,
                )

        metrics = load_evaluation_metrics(self.evaluation_dir, coin)

        if multi_out:
            names_m = list(multi_out.keys())
            weights = compute_weights_from_metrics(names_m, metrics)
            preds_per_day: list[dict[str, Any]] = []
            for d in range(HORIZON_DAYS):
                day_preds = {n: float(multi_out[n][0, d]) for n in names_m}
                w_avg = _weighted_average(day_preds, weights)
                agree = _model_agreement_score(list(day_preds.values()))
                lo, hi = _volatility_aware_bounds(
                    w_avg,
                    day_preds,
                    d + 1,
                    current_close,
                    feature_context,
                )
                preds_per_day.append({
                    "predictions": day_preds,
                    "average_prediction": w_avg,
                    "weighted_prediction": w_avg,
                    "lower_bound": lo,
                    "upper_bound": hi,
                    "model_agreement_score": agree,
                })
            final = preds_per_day[-1]
            if single_out:
                logger.info(
                    "Forecast path for %s uses %d multi-output model(s); %d single-output model(s) skipped",
                    coin, len(multi_out), len(single_out),
                )
            return {
                "coin": coin,
                "predictions": final["predictions"],
                "average_prediction": final["weighted_prediction"],
                "lower_bound": final["lower_bound"],
                "upper_bound": final["upper_bound"],
                "model_agreement_score": final["model_agreement_score"],
                "horizon_days": HORIZON_DAYS,
                "_path": preds_per_day,
                "_path_capable": True,
                "_model_weights": weights,
            }

        names_s = list(single_out.keys())
        weights = compute_weights_from_metrics(names_s, metrics)
        day_preds = {n: float(single_out[n][0, 0]) for n in names_s}
        w_avg = _weighted_average(day_preds, weights)
        lo, hi = _volatility_aware_bounds(
            w_avg,
            day_preds,
            HORIZON_DAYS,
            current_close,
            feature_context,
        )
        vals = list(day_preds.values())
        logger.info(
            "No multi-output models for %s; using %d single-output model(s) for day-14 point forecast only",
            coin, len(single_out),
        )
        return {
            "coin": coin,
            "predictions": day_preds,
            "average_prediction": w_avg,
            "lower_bound": lo,
            "upper_bound": hi,
            "model_agreement_score": _model_agreement_score(vals),
            "horizon_days": HORIZON_DAYS,
            "_path": None,
            "_path_capable": False,
            "_model_weights": weights,
        }

    def _feature_context_from_row(self, df: pd.DataFrame) -> dict[str, float]:
        keys = (
            "atr_14", "rolling_volatility_14", "rsi_14", "return_7d", "daily_return",
            "macd_histogram", "close",
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
        Load latest feature parquet, take most recent valid row, run prediction.
        Returns prediction dict (possibly with _path for multi-output) or None.
        """
        features_dir = Path(features_dir) if features_dir else settings.training.features_dir
        refresh_data_if_needed(coin, features_dir=features_dir)
        path = features_dir / f"{coin.replace(' ', '_')}_features.parquet"
        if not path.exists():
            logger.warning("Feature file not found: %s", path)
            return None
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.exception("Failed to load features for %s: %s", coin, e)
            return None
        if df.empty or len(df) < 1:
            logger.warning("Empty feature file for %s", coin)
            return None
        feat_cols = get_feature_columns(include_targets=False)
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            logger.warning("Missing feature columns for %s: %s", coin, missing[:10])
            return None
        avail = [c for c in feat_cols if c in df.columns]
        last = df[avail].iloc[-1:]
        last = last.dropna(axis=0, how="any")
        if last.empty:
            logger.warning("No valid feature row for %s (last row has NaN)", coin)
            return None
        X_df = last[avail].astype(np.float64)
        current_close = float(df["close"].iloc[-1]) if "close" in df.columns else None
        feature_context = self._feature_context_from_row(df)
        result = self.predict(coin, X_df, current_close=current_close, feature_context=feature_context)
        if not result.get("predictions"):
            return None
        result["_current_close"] = current_close
        result["_last_date"] = df["date"].iloc[-1] if "date" in df.columns else None
        result["_feature_context"] = feature_context
        return result

    def forecast_path(
        self,
        coin: str,
        features_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Produce full 14-day forecast path with dates, per-day bounds, and summary.
        Calendar dates always start tomorrow in UTC (today = UTC calendar day).
        Path prices are smoothed, sanity-checked, and blended with prior output for stability.
        """
        raw = self.predict_from_latest_features(coin, features_dir=features_dir)
        if raw is None:
            return None
        path_data = raw.get("_path")
        if not path_data or len(path_data) != HORIZON_DAYS:
            return None

        weights = raw.get("_model_weights") or {}

        # Forecast window: always anchored to current UTC date (not feature row date).
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

        raw_prices: list[float] = []
        rows: list[dict[str, Any]] = []
        for d in range(HORIZON_DAYS):
            p = path_data[d]
            fc_date = window_start + timedelta(days=d)
            fc_dt = datetime.combine(fc_date, time.min, tzinfo=timezone.utc)
            fc_ts = fc_dt.isoformat()
            w_pred = float(p.get("weighted_prediction", p["average_prediction"]))
            raw_prices.append(w_pred)
            rows.append({
                "day_index": d + 1,
                "forecast_date": fc_date.isoformat(),
                "forecast_timestamp": fc_ts,
                "predicted_price": w_pred,
                "lower_bound": float(p["lower_bound"]),
                "upper_bound": float(p["upper_bound"]),
                "model_predictions": {k: float(v) for k, v in p["predictions"].items()},
                "ensemble_prediction": w_pred,
                "agreement_score": float(p["model_agreement_score"]),
            })

        current = float(raw.get("_current_close") or 0)
        fc_ctx = raw.get("_feature_context") or {}
        agreements = [float(p["model_agreement_score"]) for p in path_data]
        mean_agree = float(np.mean(agreements)) if agreements else 0.0

        arr_raw = np.array(raw_prices, dtype=np.float64)
        anchor = current if current > 0 else None
        smoothed = smooth_path_from_anchor(arr_raw, anchor)

        prev_prices = load_last_path(self._forecast_state_dir, coin)
        final_prices = blend_path_with_previous(list(smoothed), prev_prices)

        for i, row in enumerate(rows):
            scale_model_predictions(row, raw_prices[i], final_prices[i])
            row["predicted_price"] = round(float(final_prices[i]), 6)
            row["ensemble_prediction"] = row["predicted_price"]
            row["model_predictions"] = {k: round(float(v), 6) for k, v in row["model_predictions"].items()}
            row["agreement_score"] = round(float(row["agreement_score"]), 4)

        recenter_and_widen_bounds(rows, fc_ctx, current if current > 0 else None, mean_agree)
        apply_sanity_pass(rows, mean_agree)

        forecast_path = rows

        prices = [x["predicted_price"] for x in forecast_path]
        min_p, max_p = min(prices), max(prices)
        final = forecast_path[-1]["predicted_price"]
        if current > 0:
            trend = "up" if final > current else "down" if final < current else "flat"
        else:
            trend = "flat"

        first_date = forecast_path[0]["forecast_date"] if forecast_path else None
        last_fc_date = forecast_path[-1]["forecast_date"] if forecast_path else None

        metrics = load_evaluation_metrics(self.evaluation_dir, coin)
        rv = fc_ctx.get("rolling_volatility_14")
        ret7 = fc_ctx.get("return_7d")
        prev_trend_score = load_last_trend_score(self._forecast_state_dir, coin)
        trend_label, trend_score_stored = classify_trend_label(
            current,
            float(final),
            rv,
            ret7,
            previous_trend_score=prev_trend_score,
        )

        rel_spread_path = mean_path_model_relative_spread(forecast_path)
        day14_mp = forecast_path[-1]["model_predictions"]
        if not isinstance(day14_mp, dict):
            day14_mp = path_data[-1]["predictions"]
        confidence = compute_confidence_score(
            agreements,
            day14_mp,
            metrics,
            rolling_volatility_14=rv,
            mean_path_relative_spread=rel_spread_path,
        )

        volatility_level = classify_volatility_level(rv, current if current else None)
        explanation = build_prediction_explanation(
            fc_ctx,
            trend_label,
            confidence,
            mean_agree,
            weights,
        )
        multi_horizon = build_multi_horizon(forecast_path, current)

        save_state(
            self._forecast_state_dir,
            coin,
            [float(r["predicted_price"]) for r in forecast_path],
            trend_score_stored,
        )

        msg_parts = [data_freshness_extra] if data_freshness_extra else []
        if data_freshness == "stale":
            msg_parts.append("Prediction is based on features that may not include the latest session.")
        combined_msg = " ".join(m for m in msg_parts if m) or None

        return {
            "coin": coin,
            "current_price": round(current, 6),
            "horizon_days": HORIZON_DAYS,
            "model_agreement_score": round(float(path_data[-1]["model_agreement_score"]), 4),
            "latest_market_timestamp": latest_market_ts,
            "forecast_period_start": first_date,
            "forecast_period_end": last_fc_date,
            "forecast_path": forecast_path,
            "summary": {
                "final_day_prediction": round(final, 6),
                "min_forecast_price": round(min_p, 6),
                "max_forecast_price": round(max_p, 6),
                "average_forecast_price": round(float(np.mean(prices)), 6),
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
            "data_freshness": data_freshness,
            "data_age_hours": data_age_hours,
            "data_freshness_detail": combined_msg,
            "last_refresh_time": get_last_refresh_time(coin),
        }
