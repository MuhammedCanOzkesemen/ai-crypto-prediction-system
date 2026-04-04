"""
FastAPI application for the cryptocurrency forecasting platform.

Provides health, supported coins, real model-based prediction endpoints,
and dashboard (static + chart/evaluation API).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any

from data.data_refresh import get_last_refresh_time, refresh_data_if_needed
from prediction.artifact_audit import forecast_audit_api_subset
from prediction.predictor import Predictor
from prediction.price_format import price_decimal_places, round_price
from utils.config import settings
from utils.constants import get_supported_coins_list
from utils.logging_setup import get_logger, configure_root_logger

logger = get_logger(__name__)

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
EVALUATION_DIR = settings.training.features_dir.parent / "evaluation"

# -----------------------------------------------------------------------------
# Response schemas
# -----------------------------------------------------------------------------


class CoinPredictionResponse(BaseModel):
    """Structured prediction output per coin."""

    coin: str = Field(..., description="Coin display name")
    model_predictions: dict[str, float] = Field(..., description="Per-model price predictions")
    average_prediction: float = Field(..., description="Ensemble average prediction")
    lower_bound: float = Field(..., description="Lower confidence bound (volatility-scaled)")
    upper_bound: float = Field(..., description="Upper confidence bound (volatility-scaled)")
    model_agreement_score: float = Field(..., ge=0, le=1, description="Agreement across models")
    horizon_days: int = Field(14, description="Forecast horizon in days")
    generated_at: str = Field(..., description="ISO timestamp when prediction was generated")
    forecast_quality: str = Field("degraded", description="production | degraded | legacy-fallback")
    artifact_mode: str = Field("legacy", description="production | degraded | legacy")
    schema_match: bool = False
    scaler_used: bool = False
    used_scaler: bool = False
    exact_feature_match: bool = False
    fallback_mode: bool = True
    forecast_validity: str = Field("questionable", description="invalid | questionable | valid")
    forecast_quality_score: float = Field(0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Trade-aware confidence (prediction quality × decision reliability)",
    )
    confidence_composition_reference: str = Field("", description="How confidence_score is composed")
    high_conviction: bool = Field(
        False,
        description="High confidence, agreement, signal strength, classifier aligned, vol not HIGH",
    )
    signal_strength_score: float = Field(0.0, ge=0.0, le=1.0)
    directional_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Optional down/neutral/up from directional head",
    )
    directional_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="max(up, down, neutral) from merged directional classifier probabilities",
    )
    combined_agreement_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Blend of regression path agreement and LR vs XGB directional agreement",
    )
    volatility_regime: str = Field("UNKNOWN", description="LOW | MEDIUM | HIGH from log-vol regime")
    realism_guardrail_triggered: bool = False
    forecast_audit: dict[str, Any] = Field(default_factory=dict)
    forecast_diagnostics: dict[str, Any] = Field(default_factory=dict)
    is_constant_prediction: bool = False
    low_variance_warning: bool = False
    degraded_input: bool = False
    trade_signal: str = Field("NO_TRADE", description="BUY | SELL | NO_TRADE (legacy decision layer)")
    trade_decision: str = Field(
        "NO_TRADE",
        description="Strict engine: STRONG_BUY | WEAK_BUY | NO_TRADE",
    )
    trade_reasons: list[str] = Field(
        default_factory=list,
        description="Structured reasons for trade_decision tier",
    )
    edge_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Trade engine edge: conf × agreement × signal_strength (± modifiers)",
    )
    expected_move_pct: float = Field(0.0, ge=0.0)
    expected_move_strength: str = Field("WEAK", description="WEAK | MODERATE | STRONG")
    risk_reward_ratio: float = Field(0.0, ge=0.0)
    trade_valid: bool = Field(False)
    directional_alignment: bool = Field(False)
    trend_consistency_score: float = Field(0.0, ge=0.0, le=1.0)
    decision_threshold_scale: float = Field(1.0, ge=0.5, le=1.5)
    recent_no_trade_fraction: float = Field(0.0, ge=0.0, le=1.0)
    trade_missing_for_actionable: list[str] = Field(default_factory=list)
    twitter_sentiment_used: bool = Field(
        False,
        description="True when tweet volume exceeded threshold and sentiment influenced confidence",
    )
    sentiment_alignment: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="1 = sentiment agrees with forecast direction, -1 = conflict, 0 = neutral / N/A",
    )
    sentiment_confidence_contribution: float = Field(
        0.0,
        description="Additive change to raw financial confidence from sentiment (capped)",
    )
    stability_score: float = Field(0.0, ge=0.0, le=1.0, description="Forecast path stability [0,1]")
    consensus_score: float = Field(0.0, ge=0.0, le=1.0, description="Multi-day ensemble direction consensus")
    trend_confirmation_score: float = Field(
        0.0, ge=0.0, le=1.0, description="EMA/ADX trend confirmation for long bias"
    )
    volatility_shock_detected: bool = Field(
        False,
        description="True when short-term vol exceeds ~1.8× recent average",
    )
    market_regime: str = Field(
        "RANGING",
        description="TRENDING | RANGING | VOLATILE (structure + vol context)",
    )
    market_regime_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Classifier confidence in the current regime label",
    )


class MarketScannerRow(BaseModel):
    """Compact row for the all-coins market scanner."""

    coin: str
    directional_confidence: float = Field(0.0, ge=0.0, le=1.0)
    combined_agreement_score: float = Field(0.0, ge=0.0, le=1.0)
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    trade_decision: str = Field("NO_TRADE")
    expected_move_pct: float = Field(0.0, ge=0.0)
    volatility_regime: str = Field("UNKNOWN")
    signal_strength_score: float = Field(0.0, ge=0.0, le=1.0)
    directional_probabilities: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"


class ForecastDayItem(BaseModel):
    """Single day in forecast path."""
    day_index: int
    forecast_date: str
    forecast_timestamp: str | None = None
    predicted_price: float
    predicted_log_return: float | None = None
    predicted_return: float | None = None
    volatility: float | None = None
    lower_bound: float
    upper_bound: float
    model_predictions: dict[str, float]
    ensemble_prediction: float
    agreement_score: float


class ForecastPathSummary(BaseModel):
    """Summary of 14-day forecast path."""
    final_day_prediction: float
    min_forecast_price: float
    max_forecast_price: float
    average_forecast_price: float
    trend_direction_14d: str
    trend_label: str = Field(
        "NEUTRAL",
        description="STRONG UP | UP | NEUTRAL | DOWN | STRONG DOWN",
    )


class HorizonSnapshot(BaseModel):
    """Single-horizon slice (1d / 3d / 7d / 14d)."""
    day_index: int
    forecast_date: str | None = None
    predicted_price: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    implied_return_vs_spot: float | None = None
    agreement_score: float | None = None


def _raise_if_prediction_unavailable(
    predictor: Predictor,
    match: str,
    features_dir: Path,
) -> None:
    """
    Raise HTTPException with structured detail when forecast/predictions fail.
    Scaler missing alone does not trigger this (inference falls back to raw features).
    """
    features_path = features_dir / f"{match.replace(' ', '_')}_features.parquet"
    err = predictor.get_last_forecast_error() or {}
    code = str(err.get("code", "prediction_failed"))
    message = str(err.get("message", "Prediction could not be completed."))
    hint = err.get("hint")
    detail: dict[str, str] = {
        "error": code,
        "message": message,
        "coin": match,
    }
    if hint is not None and str(hint).strip():
        detail["hint"] = str(hint)
    if not features_path.exists():
        detail["error"] = "no_feature_file"
        detail["message"] = f"Feature file not found: {features_path.name}"
        detail["hint"] = (
            f"Run `python scripts/train_pipeline.py --coin {match}` or POST /api/refresh/{match}"
        )
        raise HTTPException(status_code=404, detail=detail)
    raise HTTPException(status_code=503, detail=detail)


def _data_freshness(latest_market_ts: str | None) -> tuple[str, str | None]:
    """
    Fallback freshness if predictor did not attach fields.
    Stale if market data is older than 24 hours (UTC).
    """
    if not latest_market_ts or not latest_market_ts.strip():
        return "stale", "Market data date unknown."
    try:
        ts = pd.to_datetime(latest_market_ts)
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        now = datetime.now(timezone.utc)
        age_hours = (now - ts).total_seconds() / 3600.0
        if age_hours <= 24.0:
            return "fresh", None
        return "stale", f"Underlying market data is {int(age_hours)} hours old. Refresh features for latest data."
    except Exception:
        return "stale", "Could not determine data freshness."


def _scanner_sort_key(row: MarketScannerRow) -> tuple[int, float, float, float, str]:
    pri = {"STRONG_BUY": 0, "WEAK_BUY": 1, "NO_TRADE": 2}
    return (
        pri.get(str(row.trade_decision).upper(), 3),
        -float(row.confidence_score),
        -float(row.expected_move_pct),
        -float(row.combined_agreement_score),
        str(row.coin),
    )


class ForecastPathResponse(BaseModel):
    """Full 14-day forecast path with summary and intelligence layer."""
    coin: str
    current_price: float
    horizon_days: int
    generated_at: str
    latest_market_timestamp: str | None = None
    forecast_period_start: str | None = None
    forecast_period_end: str | None = None
    data_freshness: str = "unknown"
    data_freshness_message: str | None = None
    data_age_hours: int | None = Field(
        None,
        description="Whole hours since latest_market_timestamp (UTC); None if unknown",
    )
    last_refresh_time: str | None = Field(
        None,
        description="ISO UTC timestamp of last successful data/feature refresh for this coin",
    )
    forecast_path: list[ForecastDayItem]
    summary: ForecastPathSummary
    evaluation: dict | None = None
    confidence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Trade-aware confidence (prediction quality × decision reliability)",
    )
    trend_label: str = Field("NEUTRAL", description="Classified 14d trend vs spot (path-based)")
    explanation: str = Field("", description="Human-readable forecast rationale")
    model_weights: dict[str, float] = Field(default_factory=dict, description="Metric-weighted ensemble weights")
    volatility_level: str = Field("UNKNOWN", description="LOW | MEDIUM | HIGH | UNKNOWN")
    multi_horizon: dict[str, HorizonSnapshot] = Field(
        default_factory=dict,
        description="Snapshots at 1d, 3d, 7d, 14d",
    )
    mean_path_agreement: float = Field(0.0, ge=0.0, le=1.0, description="Mean daily model agreement")
    model_agreement_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Model agreement on final horizon day (day 14)",
    )
    forecast_quality: str = Field(
        "degraded",
        description="production | degraded | legacy-fallback",
    )
    artifact_mode: str = Field("legacy", description="production | degraded | legacy")
    schema_match: bool = Field(False, description="Disk schema + columns + live feature index aligned")
    scaler_used: bool = Field(False, description="Scaler bundle applied successfully at inference")
    used_scaler: bool = Field(False, description="Whether RobustScaler bundle was applied")
    exact_feature_match: bool = Field(
        False,
        description="Training feature names fully present; no emergency width align",
    )
    fallback_mode: bool = Field(
        True,
        description="True when not full production artifact parity",
    )
    forecast_validity: str = Field("questionable", description="invalid | questionable | valid")
    forecast_quality_score: float = Field(0.0, ge=0.0, le=1.0)
    confidence_composition_reference: str = Field(
        "",
        description="How confidence_score is composed (documentation string)",
    )
    realism_guardrail_triggered: bool = Field(
        False,
        description="Cumulative log-return path was scaled to a volatility/historical cap",
    )
    forecast_audit: dict[str, Any] = Field(
        default_factory=dict,
        description="Artifact tier, schema versions, feature match ratio",
    )
    forecast_diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Sanity check, clip saturation, align flags",
    )
    is_constant_prediction: bool = Field(
        False,
        description="True if path is flat (invalid as directional forecast)",
    )
    low_variance_warning: bool = Field(False, description="Unusually low path variance")
    degraded_input: bool = Field(False, description="High missing-feature fill or bad align")
    high_conviction: bool = Field(
        False,
        description="High confidence, agreement, signal strength, classifier aligned, vol not HIGH",
    )
    signal_strength_score: float = Field(0.0, ge=0.0, le=1.0)
    directional_probabilities: dict[str, float] = Field(default_factory=dict)
    directional_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="max(up, down, neutral) from merged directional probabilities",
    )
    combined_agreement_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Regression agreement blended with directional-model agreement",
    )
    volatility_regime: str = Field("UNKNOWN", description="LOW | MEDIUM | HIGH | UNKNOWN")
    trade_signal: str = Field("NO_TRADE", description="BUY | SELL | NO_TRADE (legacy decision filter)")
    trade_decision: str = Field(
        "NO_TRADE",
        description="Strict engine: STRONG_BUY | WEAK_BUY | NO_TRADE",
    )
    trade_reasons: list[str] = Field(
        default_factory=list,
        description="Structured reasons for trade_decision tier",
    )
    edge_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Trade engine edge: conf × agreement × signal_strength (± modifiers)",
    )
    expected_move_pct: float = Field(0.0, ge=0.0, description="|final−spot|/spot")
    expected_move_strength: str = Field("WEAK", description="WEAK | MODERATE | STRONG")
    risk_reward_ratio: float = Field(0.0, ge=0.0, description="|predicted move| / (upper−lower)")
    trade_valid: bool = Field(False, description="True only if filter allows BUY or SELL")
    directional_alignment: bool = Field(
        False,
        description="14d trend sign matches forecast move and classifier agrees",
    )
    trend_consistency_score: float = Field(0.0, ge=0.0, le=1.0)
    decision_threshold_scale: float = Field(1.0, ge=0.5, le=1.5)
    recent_no_trade_fraction: float = Field(0.0, ge=0.0, le=1.0)
    trade_missing_for_actionable: list[str] = Field(default_factory=list)
    twitter_sentiment_used: bool = Field(
        False,
        description="True when tweet volume exceeded threshold and sentiment influenced confidence",
    )
    sentiment_alignment: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="1 = sentiment agrees with forecast direction, -1 = conflict, 0 = neutral / N/A",
    )
    sentiment_confidence_contribution: float = Field(
        0.0,
        description="Additive change to raw financial confidence from sentiment (capped)",
    )
    stability_score: float = Field(0.0, ge=0.0, le=1.0, description="Forecast path stability [0,1]")
    consensus_score: float = Field(0.0, ge=0.0, le=1.0, description="Multi-day ensemble direction consensus")
    trend_confirmation_score: float = Field(
        0.0, ge=0.0, le=1.0, description="EMA/ADX trend confirmation for long bias"
    )
    volatility_shock_detected: bool = Field(
        False,
        description="True when short-term vol exceeds ~1.8× recent average",
    )
    market_regime: str = Field(
        "RANGING",
        description="TRENDING | RANGING | VOLATILE (structure + vol context)",
    )
    market_regime_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Classifier confidence in the current regime label",
    )


class RefreshResponse(BaseModel):
    """Result of POST /api/refresh/{coin}."""
    coin: str
    refreshed: bool = Field(..., description="True if fetch + feature rebuild ran successfully")
    last_refresh_time: str | None = None
    detail: str | None = None


# -----------------------------------------------------------------------------
# App factory and routes
# -----------------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        debug=settings.api.debug,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    predictor = Predictor()

    @app.on_event("startup")
    def startup() -> None:
        configure_root_logger()
        logger.info("API starting: %s", settings.api.title)
        # Fail fast if forecast-path route is missing (prevents stale/cached app)
        paths = [getattr(r, "path", "") for r in app.routes]
        if "/api/forecast-path/{coin}" not in paths:
            raise RuntimeError(
                "Route /api/forecast-path/{coin} not registered. "
                "Clear __pycache__, kill all uvicorn processes, and restart."
            )

    @app.get("/")
    def root() -> dict[str, str]:
        """Root route; confirms API is running."""
        return {"message": "Crypto Forecasting API is running"}

    # Debug routes (temporary - remove after fixing dashboard)
    @app.get("/debug/routes", include_in_schema=False)
    def debug_routes():
        """Return all registered routes for diagnostics."""
        routes = []
        for r in app.routes:
            p = getattr(r, "path", None)
            methods = getattr(r, "methods", None)
            name = getattr(r, "name", None)
            routes.append({"path": p, "methods": list(methods) if methods else None, "name": name})
        return {"routes": routes, "app_module": "api.app:app"}

    @app.get("/debug/dashboard-path", include_in_schema=False)
    def debug_dashboard_path():
        """Return dashboard path diagnostics."""
        index_path = DASHBOARD_DIR / "index.html"
        static_dir = DASHBOARD_DIR / "static"
        return {
            "DASHBOARD_DIR": str(DASHBOARD_DIR),
            "DASHBOARD_DIR_exists": DASHBOARD_DIR.exists(),
            "index_path": str(index_path),
            "index_exists": index_path.exists(),
            "static_dir_exists": static_dir.exists(),
            "__file__": str(Path(__file__).resolve()),
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        """Quiet favicon handling to avoid 404 logging."""
        from fastapi.responses import Response
        return Response(status_code=204)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Health check for load balancers and monitoring."""
        return HealthResponse(status="ok", version=settings.api.version)

    @app.get("/coins")
    def coins() -> dict[str, list[str]]:
        """Return list of supported coins (display names)."""
        return {"coins": get_supported_coins_list()}

    # API router: /api/* routes (forecast-path, chart, evaluation) - all appear in Swagger
    api_router = APIRouter(tags=["api"])

    @api_router.get("/forecast-path/{coin}", response_model=ForecastPathResponse)
    def forecast_path(
        coin: str = PathParam(..., description="Coin display name"),
    ) -> ForecastPathResponse:
        """
        Get full 14-day forecast path with per-day predictions, bounds, and summary.
        Returns 503 if only legacy single-output models exist (retrain for path support).
        """
        supported = get_supported_coins_list()
        match = next((c for c in supported if c.lower() == coin.strip().lower()), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin}")
        features_dir = settings.training.features_dir
        path_result = predictor.forecast_path(match, features_dir=features_dir)
        if path_result is None:
            _raise_if_prediction_unavailable(predictor, match, features_dir)
        now = datetime.now(timezone.utc).isoformat()
        latest_ts = path_result.get("latest_market_timestamp")
        freshness_status = str(path_result.get("data_freshness") or "unknown")
        data_age_hours = path_result.get("data_age_hours")
        freshness_message = path_result.get("data_freshness_detail")
        if freshness_status not in ("fresh", "stale") or freshness_message is None:
            fb_status, fb_msg = _data_freshness(latest_ts)
            if freshness_status not in ("fresh", "stale"):
                freshness_status = fb_status
            if freshness_message is None:
                freshness_message = fb_msg
        eval_path = EVALUATION_DIR / f"{match.replace(' ', '_')}_metrics.json"
        evaluation = None
        if eval_path.exists():
            try:
                eval_data = json.loads(eval_path.read_text())
                best = min(eval_data.items(), key=lambda x: x[1].get("mae", float("inf")))
                evaluation = {
                    "best_model": best[0].replace("_", " ").title(),
                    "mae": round(best[1].get("mae", 0), 4),
                    "rmse": round(best[1].get("rmse", 0), 4),
                    "mape_pct": round(best[1].get("mape", 0) * 100, 2),
                    "directional_accuracy_pct": round(best[1].get("directional_accuracy", 0) * 100, 2),
                }
            except Exception:
                pass
        mh_raw = path_result.get("multi_horizon") or {}
        multi_horizon: dict[str, HorizonSnapshot] = {}
        for key, snap in mh_raw.items():
            if isinstance(snap, dict):
                multi_horizon[str(key)] = HorizonSnapshot(**snap)

        return ForecastPathResponse(
            coin=path_result["coin"],
            current_price=path_result["current_price"],
            horizon_days=path_result["horizon_days"],
            generated_at=now,
            latest_market_timestamp=latest_ts,
            forecast_period_start=path_result.get("forecast_period_start"),
            forecast_period_end=path_result.get("forecast_period_end"),
            data_freshness=freshness_status,
            data_freshness_message=freshness_message,
            data_age_hours=int(data_age_hours) if data_age_hours is not None else None,
            last_refresh_time=path_result.get("last_refresh_time"),
            forecast_path=[ForecastDayItem(**d) for d in path_result["forecast_path"]],
            summary=ForecastPathSummary(**path_result["summary"]),
            evaluation=evaluation,
            confidence_score=float(path_result.get("confidence_score", 0.0)),
            trend_label=str(path_result.get("trend_label") or "NEUTRAL"),
            explanation=str(path_result.get("explanation", "")),
            model_weights=dict(path_result.get("model_weights") or {}),
            volatility_level=str(path_result.get("volatility_level", "UNKNOWN")),
            multi_horizon=multi_horizon,
            mean_path_agreement=float(path_result.get("mean_path_agreement", 0.0)),
            model_agreement_score=float(path_result.get("model_agreement_score", 0.0)),
            forecast_quality=str(path_result.get("forecast_quality", "degraded")),
            artifact_mode=str(path_result.get("artifact_mode", "legacy")),
            schema_match=bool(path_result.get("schema_match", False)),
            scaler_used=bool(path_result.get("scaler_used", False)),
            used_scaler=bool(path_result.get("used_scaler", False)),
            exact_feature_match=bool(path_result.get("exact_feature_match", False)),
            fallback_mode=bool(path_result.get("fallback_mode", True)),
            forecast_validity=str(path_result.get("forecast_validity", "questionable")),
            forecast_quality_score=float(path_result.get("forecast_quality_score") or 0.0),
            confidence_composition_reference=str(
                path_result.get("confidence_composition_reference") or ""
            ),
            realism_guardrail_triggered=bool(path_result.get("realism_guardrail_triggered", False)),
            forecast_audit=dict(path_result.get("forecast_audit") or {}),
            forecast_diagnostics=dict(path_result.get("forecast_diagnostics") or {}),
            is_constant_prediction=bool(path_result.get("is_constant_prediction", False)),
            low_variance_warning=bool(path_result.get("low_variance_warning", False)),
            degraded_input=bool(path_result.get("degraded_input", False)),
            high_conviction=bool(path_result.get("high_conviction", False)),
            signal_strength_score=float(path_result.get("signal_strength_score") or 0.0),
            directional_probabilities=dict(path_result.get("directional_probabilities") or {}),
            directional_confidence=float(path_result.get("directional_confidence") or 0.0),
            combined_agreement_score=float(path_result.get("combined_agreement_score") or 0.0),
            volatility_regime=str(path_result.get("volatility_regime") or "UNKNOWN"),
            trade_signal=str(path_result.get("trade_signal") or "NO_TRADE"),
            trade_decision=str(path_result.get("trade_decision", "NO_TRADE")),
            trade_reasons=list(path_result.get("trade_reasons") or []),
            edge_score=float(path_result.get("edge_score") or 0.0),
            expected_move_pct=float(path_result.get("expected_move_pct") or 0.0),
            expected_move_strength=str(path_result.get("expected_move_strength") or "WEAK"),
            risk_reward_ratio=float(path_result.get("risk_reward_ratio") or 0.0),
            trade_valid=bool(path_result.get("trade_valid", False)),
            directional_alignment=bool(path_result.get("directional_alignment", False)),
            trend_consistency_score=float(path_result.get("trend_consistency_score") or 0.0),
            decision_threshold_scale=float(path_result.get("decision_threshold_scale") or 1.0),
            recent_no_trade_fraction=float(path_result.get("recent_no_trade_fraction") or 0.0),
            trade_missing_for_actionable=list(path_result.get("trade_missing_for_actionable") or []),
            twitter_sentiment_used=bool(
                (path_result.get("forecast_diagnostics") or {}).get("twitter_sentiment_used", False)
            ),
            sentiment_alignment=float(
                (path_result.get("forecast_diagnostics") or {}).get("sentiment_alignment", 0.0)
            ),
            sentiment_confidence_contribution=float(
                (path_result.get("forecast_diagnostics") or {}).get(
                    "sentiment_confidence_contribution", 0.0
                )
            ),
            stability_score=float(path_result.get("stability_score", 0.0)),
            consensus_score=float(path_result.get("consensus_score", 0.0)),
            trend_confirmation_score=float(path_result.get("trend_confirmation_score", 0.0)),
            volatility_shock_detected=bool(path_result.get("volatility_shock_detected", False)),
            market_regime=str(path_result.get("market_regime", "RANGING")),
            market_regime_confidence=float(path_result.get("market_regime_confidence", 0.0)),
        )

    @api_router.post("/refresh/{coin}", response_model=RefreshResponse)
    def force_refresh_coin(
        coin: str = PathParam(..., description="Coin display name"),
    ) -> RefreshResponse:
        """Force OHLCV fetch and feature parquet rebuild for one coin."""
        supported = get_supported_coins_list()
        match = next((c for c in supported if c.lower() == coin.strip().lower()), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin}")
        features_dir = settings.training.features_dir
        ok = refresh_data_if_needed(match, force=True, features_dir=features_dir)
        ts = get_last_refresh_time(match) if ok else None
        return RefreshResponse(
            coin=match,
            refreshed=ok,
            last_refresh_time=ts,
            detail=None if ok else "Refresh failed (fetch or feature build). Check logs.",
        )

    @api_router.get("/chart/{coin}")
    def chart_data(
        coin: str = PathParam(..., description="Coin display name"),
        days: int = 90,
    ):
        """Return historical close prices for charting. Reads from feature parquet."""
        supported = get_supported_coins_list()
        match = next((c for c in supported if c.lower() == coin.strip().lower()), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin}")
        path = settings.training.features_dir / f"{match.replace(' ', '_')}_features.parquet"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"No feature data for {match}")
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.exception("Failed to load features for chart: %s", e)
            raise HTTPException(status_code=500, detail="Failed to load chart data")
        if "date" not in df.columns or "close" not in df.columns:
            raise HTTPException(status_code=500, detail="Feature file missing date/close columns")
        df = df[["date", "close"]].dropna().tail(days)
        dates = df["date"].astype(str).tolist()
        ref = float(df["close"].iloc[-1]) if len(df) > 0 else 1.0
        d = price_decimal_places(ref)
        closes = [round(float(c), d) for c in df["close"].astype(float)]
        last_date = df["date"].iloc[-1] if len(df) > 0 else None
        latest_market_date = None
        if last_date is not None:
            try:
                latest_market_date = pd.to_datetime(last_date).isoformat()
            except Exception:
                latest_market_date = str(last_date)
        return {
            "coin": match,
            "dates": dates,
            "close": closes,
            "latest_market_date": latest_market_date,
        }

    @api_router.get("/evaluation/{coin}")
    def evaluation_summary(
        coin: str = PathParam(..., description="Coin display name"),
    ):
        """Return evaluation metrics for a coin. Best model = lowest MAE."""
        supported = get_supported_coins_list()
        match = next((c for c in supported if c.lower() == coin.strip().lower()), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin}")
        path = EVALUATION_DIR / f"{match.replace(' ', '_')}_metrics.json"
        if not path.exists():
            return {"coin": match, "best_model": None, "metrics": None}
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            logger.exception("Failed to load evaluation for %s: %s", match, e)
            return {"coin": match, "best_model": None, "metrics": None}
        best = min(data.items(), key=lambda x: x[1].get("mae", float("inf")))
        best_model = best[0].replace("_", " ").title()
        metrics = best[1]
        return {
            "coin": match,
            "best_model": best_model,
            "metrics": {
                "mae": round(metrics.get("mae", 0), 4),
                "rmse": round(metrics.get("rmse", 0), 4),
                "mape": round(metrics.get("mape", 0) * 100, 2),
                "directional_accuracy": round(metrics.get("directional_accuracy", 0) * 100, 2),
            },
        }

    @api_router.get("/backtest/{coin}")
    def backtest_saved_summary(
        coin: str = PathParam(..., description="Coin display name"),
    ) -> dict[str, Any]:
        """
        Return the latest saved trade backtest summary JSON from ``artifacts/backtests/``.

        Does not run a backtest on request; run ``python scripts/run_backtest.py`` offline first.
        """
        supported = get_supported_coins_list()
        match = next((c for c in supported if c.lower() == coin.strip().lower()), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Unsupported coin: {coin}")
        path = settings.data.artifact_dir / "backtests" / f"{match.replace(' ', '_')}_summary.json"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="No backtest summary on disk. Run: python scripts/run_backtest.py --coin " + match,
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("Failed to read backtest summary for %s: %s", match, e)
            raise HTTPException(status_code=500, detail="Backtest summary file is unreadable") from e

    app.include_router(api_router, prefix="/api")

    @app.get("/predictions/{coin}", response_model=CoinPredictionResponse)
    def predictions(
        coin: str = PathParam(..., description="Coin display name (e.g. Bitcoin, Ethereum)"),
    ) -> CoinPredictionResponse:
        """
        Get 14-day price forecast for a coin using trained models.
        Loads latest feature parquet and runs ensemble prediction.
        """
        supported = get_supported_coins_list()
        normalized = coin.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="Coin name cannot be empty")
        match = next((c for c in supported if c.lower() == normalized.lower()), None)
        if match is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unsupported coin: '{coin}'. Supported: {supported}",
            )
        features_dir = settings.training.features_dir
        result = predictor.predict_from_latest_features(match, features_dir=features_dir)
        if result is None:
            _raise_if_prediction_unavailable(predictor, match, features_dir)
        now = datetime.now(timezone.utc).isoformat()
        diag = result.get("_forecast_diagnostics") or {}
        cref = float(result.get("_current_close") or 0)
        pref = cref if cref > 0 else max(abs(float(result["average_prediction"])), 1e-30)
        return CoinPredictionResponse(
            coin=result["coin"],
            model_predictions={k: round_price(float(v), pref) for k, v in result["predictions"].items()},
            average_prediction=round_price(float(result["average_prediction"]), pref),
            lower_bound=round_price(float(result["lower_bound"]), pref),
            upper_bound=round_price(float(result["upper_bound"]), pref),
            model_agreement_score=result["model_agreement_score"],
            horizon_days=result["horizon_days"],
            generated_at=now,
            forecast_quality=str(diag.get("forecast_quality", "degraded")),
            artifact_mode=str(diag.get("artifact_mode", "legacy")),
            schema_match=bool(diag.get("schema_match", False)),
            scaler_used=bool(diag.get("used_scaler", False)),
            used_scaler=bool(diag.get("used_scaler", False)),
            exact_feature_match=bool(diag.get("exact_feature_match", False)),
            fallback_mode=bool(diag.get("fallback_mode", True)),
            forecast_validity=str(diag.get("forecast_validity", "questionable")),
            forecast_quality_score=float(diag.get("forecast_quality_score") or 0.0),
            confidence_score=float(diag.get("confidence_score") or 0.0),
            confidence_composition_reference=str(diag.get("confidence_composition_reference") or ""),
            high_conviction=bool(diag.get("high_conviction", False)),
            signal_strength_score=float(diag.get("signal_strength_score") or 0.0),
            directional_probabilities=dict(diag.get("directional_probabilities") or {}),
            directional_confidence=float(diag.get("directional_confidence") or 0.0),
            combined_agreement_score=float(diag.get("combined_agreement_score") or 0.0),
            volatility_regime=str(diag.get("volatility_regime") or "UNKNOWN"),
            realism_guardrail_triggered=bool(diag.get("realism_guardrail_cumulative", False)),
            forecast_audit=forecast_audit_api_subset(diag.get("forecast_audit") or {}),
            forecast_diagnostics={
                "used_emergency_width_align": bool(diag.get("used_emergency_width_align")),
                "scaler_transform_failed": bool(diag.get("scaler_transform_failed")),
                "clip_saturation_step_fraction": float(diag.get("clip_saturation_step_fraction", 0.0)),
                "sanity_check": diag.get("sanity_check") or {},
                "step0_raw_logret_by_model": diag.get("step0_raw_logret_by_model") or {},
                "model_horizon_variance_raw": diag.get("model_horizon_variance_raw") or {},
                "agreement_diagnostics": diag.get("agreement_diagnostics") or {},
                "max_missing_feature_fill_ratio": float(diag.get("max_missing_feature_fill_ratio", 0.0)),
                "twitter_sentiment_used": bool(diag.get("twitter_sentiment_used", False)),
                "sentiment_alignment": float(diag.get("sentiment_alignment", 0.0)),
                "sentiment_confidence_contribution": float(
                    diag.get("sentiment_confidence_contribution", 0.0)
                ),
                "consensus_score": float(diag.get("consensus_score", 0.0)),
                "stability_score": float(diag.get("stability_score", 0.0)),
                "trend_confirmation_score": float(diag.get("trend_confirmation_score", 0.0)),
                "volatility_shock_detected": bool(diag.get("volatility_shock_detected", False)),
            },
            is_constant_prediction=bool(diag.get("is_constant_prediction", False)),
            low_variance_warning=bool(diag.get("low_variance_warning", False)),
            degraded_input=bool(diag.get("degraded_input", False)),
            trade_signal=str(diag.get("trade_signal") or "NO_TRADE"),
            trade_decision=str(diag.get("trade_decision", "NO_TRADE")),
            trade_reasons=list(diag.get("trade_reasons") or []),
            edge_score=float(diag.get("edge_score") or 0.0),
            expected_move_pct=float(diag.get("expected_move_pct") or 0.0),
            expected_move_strength=str(diag.get("expected_move_strength") or "WEAK"),
            risk_reward_ratio=float(diag.get("risk_reward_ratio") or 0.0),
            trade_valid=bool(diag.get("trade_valid", False)),
            directional_alignment=bool(diag.get("directional_alignment", False)),
            trend_consistency_score=float(diag.get("trend_consistency_score") or 0.0),
            decision_threshold_scale=float(diag.get("decision_threshold_scale") or 1.0),
            recent_no_trade_fraction=float(diag.get("recent_no_trade_fraction") or 0.0),
            trade_missing_for_actionable=list(diag.get("trade_missing_for_actionable") or []),
            twitter_sentiment_used=bool(diag.get("twitter_sentiment_used", False)),
            sentiment_alignment=float(diag.get("sentiment_alignment", 0.0)),
            sentiment_confidence_contribution=float(diag.get("sentiment_confidence_contribution", 0.0)),
            stability_score=float(diag.get("stability_score", 0.0)),
            consensus_score=float(diag.get("consensus_score", 0.0)),
            trend_confirmation_score=float(diag.get("trend_confirmation_score", 0.0)),
            volatility_shock_detected=bool(diag.get("volatility_shock_detected", False)),
            market_regime=str(diag.get("market_regime", "RANGING")),
            market_regime_confidence=float(diag.get("market_regime_confidence", 0.0)),
        )

    @app.get("/api/predictions/all", response_model=list[MarketScannerRow])
    def predictions_all() -> list[MarketScannerRow]:
        """Compact scanner rows for all supported coins, sorted by trade priority."""
        features_dir = settings.training.features_dir
        rows: list[MarketScannerRow] = []
        for coin in get_supported_coins_list():
            try:
                result = predictor.predict_from_latest_features(coin, features_dir=features_dir)
            except Exception as e:
                logger.warning("Scanner prediction failed for %s: %s", coin, e)
                continue
            if not result:
                continue
            diag = result.get("_forecast_diagnostics") or {}
            rows.append(
                MarketScannerRow(
                    coin=str(result.get("coin") or coin),
                    directional_confidence=float(diag.get("directional_confidence") or 0.0),
                    combined_agreement_score=float(diag.get("combined_agreement_score") or 0.0),
                    confidence_score=float(diag.get("confidence_score") or 0.0),
                    trade_decision=str(diag.get("trade_decision") or "NO_TRADE"),
                    expected_move_pct=float(diag.get("expected_move_pct") or 0.0),
                    volatility_regime=str(diag.get("volatility_regime") or "UNKNOWN"),
                    signal_strength_score=float(diag.get("signal_strength_score") or 0.0),
                    directional_probabilities=dict(diag.get("directional_probabilities") or {}),
                )
            )
        rows.sort(key=_scanner_sort_key)
        return rows

    # Dashboard (defined before mount so it takes precedence)
    index_path = DASHBOARD_DIR / "index.html"

    @app.get("/dashboard", include_in_schema=False)
    @app.get("/dashboard/", include_in_schema=False)
    def dashboard():
        """Serve the dashboard UI."""
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return FileResponse(str(index_path), media_type="text/html")

    # Static assets (mount after explicit routes so /dashboard and /api/* take precedence)
    static_dir = DASHBOARD_DIR / "static"
    if static_dir.exists():
        app.mount("/dashboard/static", StaticFiles(directory=str(static_dir)), name="dashboard_static")

    return app


app = create_app()
