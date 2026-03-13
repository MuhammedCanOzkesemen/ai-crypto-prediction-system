"""
Prediction service: loads trained models and produces structured forecasts.

Supports both single-point (legacy) and 14-day forecast path output.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.feature_builder import get_feature_columns
from models.model_registry import get_model, list_models
from utils.config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

HORIZON_DAYS = 14


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


def _bounds_from_predictions(predictions: list[float]) -> tuple[float, float]:
    """Lower and upper bounds from ensemble predictions."""
    if not predictions:
        return 0.0, 0.0
    arr = np.array(predictions)
    return float(arr.min()), float(arr.max())


class Predictor:
    """
    Loads trained models per coin and runs ensemble predictions.
    Supports multi-output (14-day path) and falls back to single-output for legacy models.
    """

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = Path(models_dir) if models_dir else settings.training.models_dir
        self._loaded: dict[str, dict[str, Any]] = {}  # coin -> {model_name -> model}

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

        # Collect per-model outputs; validate shape per model (mix of single/multi possible)
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

        # Split by output dim: multi-output (14 steps) vs single-output (legacy)
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

        if multi_out:
            # Build 14-day path from multi-output models only
            preds_per_day: list[dict[str, float]] = []
            for d in range(HORIZON_DAYS):
                day_vals = [float(multi_out[n][0, d]) for n in multi_out]
                preds_per_day.append({
                    "predictions": {n: float(multi_out[n][0, d]) for n in multi_out},
                    "average_prediction": float(np.mean(day_vals)),
                    "lower_bound": float(min(day_vals)),
                    "upper_bound": float(max(day_vals)),
                    "model_agreement_score": _model_agreement_score(day_vals),
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
                "average_prediction": final["average_prediction"],
                "lower_bound": final["lower_bound"],
                "upper_bound": final["upper_bound"],
                "model_agreement_score": final["model_agreement_score"],
                "horizon_days": HORIZON_DAYS,
                "_path": preds_per_day,
                "_path_capable": True,
            }

        # Only single-output models: legacy single-point
        vals = [float(single_out[n][0, 0]) for n in single_out]
        logger.info(
            "No multi-output models for %s; using %d single-output model(s) for day-14 point forecast only",
            coin, len(single_out),
        )
        return {
            "coin": coin,
            "predictions": {n: float(v) for n, v in zip(single_out, vals)},
            "average_prediction": float(np.mean(vals)),
            "lower_bound": float(min(vals)),
            "upper_bound": float(max(vals)),
            "model_agreement_score": _model_agreement_score(vals),
            "horizon_days": HORIZON_DAYS,
            "_path": None,
            "_path_capable": False,
        }

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
        result = self.predict(coin, X_df, current_close=current_close)
        if not result.get("predictions"):
            return None
        result["_current_close"] = current_close
        result["_last_date"] = df["date"].iloc[-1] if "date" in df.columns else None
        return result

    def forecast_path(
        self,
        coin: str,
        features_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        """
        Produce full 14-day forecast path with dates, per-day bounds, and summary.
        Returns None if prediction fails or models are single-output (no path).
        """
        raw = self.predict_from_latest_features(coin, features_dir=features_dir)
        if raw is None:
            return None
        path_data = raw.get("_path")
        if not path_data or len(path_data) != HORIZON_DAYS:
            return None

        # Rolling user-facing window: next 14 days from current date (UTC), not artifact date
        today_utc = datetime.now(timezone.utc).date()
        window_start = today_utc + timedelta(days=1)  # tomorrow = day 1

        latest_market_ts: str | None = None
        last_date = raw.get("_last_date")
        if last_date is not None:
            try:
                ts = pd.to_datetime(last_date)
                latest_market_ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            except Exception:
                latest_market_ts = str(last_date) if last_date else None

        forecast_path: list[dict[str, Any]] = []
        for d in range(HORIZON_DAYS):
            p = path_data[d]
            fc_date = window_start + timedelta(days=d)
            fc_dt = datetime.combine(fc_date, time.min, tzinfo=timezone.utc)
            fc_ts = fc_dt.isoformat()
            forecast_path.append({
                "day_index": d + 1,
                "forecast_date": fc_date.isoformat(),
                "forecast_timestamp": fc_ts,
                "predicted_price": round(float(p["average_prediction"]), 6),
                "lower_bound": round(float(p["lower_bound"]), 6),
                "upper_bound": round(float(p["upper_bound"]), 6),
                "model_predictions": {k: round(float(v), 6) for k, v in p["predictions"].items()},
                "ensemble_prediction": round(float(p["average_prediction"]), 6),
                "agreement_score": round(float(p["model_agreement_score"]), 4),
            })

        prices = [x["predicted_price"] for x in forecast_path]
        min_p, max_p = min(prices), max(prices)
        final = forecast_path[-1]["predicted_price"]
        current = raw.get("_current_close")
        if current is not None and current > 0:
            trend = "up" if final > current else "down" if final < current else "flat"
        else:
            trend = "flat"

        first_date = forecast_path[0]["forecast_date"] if forecast_path else None
        last_fc_date = forecast_path[-1]["forecast_date"] if forecast_path else None

        return {
            "coin": coin,
            "current_price": round(float(current or 0), 6),
            "horizon_days": HORIZON_DAYS,
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
            },
        }
