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

from data.data_refresh import get_last_refresh_time, refresh_data_if_needed
from prediction.predictor import Predictor
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
    lower_bound: float = Field(..., description="Lower bound (min of model predictions)")
    upper_bound: float = Field(..., description="Upper bound (max of model predictions)")
    model_agreement_score: float = Field(..., ge=0, le=1, description="Agreement across models")
    horizon_days: int = Field(14, description="Forecast horizon in days")
    generated_at: str = Field(..., description="ISO timestamp when prediction was generated")


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
        "SIDEWAYS",
        description="STRONG UP | UP | SIDEWAYS | DOWN | STRONG DOWN",
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
    confidence_score: float = Field(0.0, ge=0.0, le=0.85, description="Composite AI confidence (capped)")
    trend_label: str = Field("SIDEWAYS", description="Classified trend vs spot and volatility")
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
            features_path = features_dir / f"{match.replace(' ', '_')}_features.parquet"
            if not features_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Feature file not found for {match}. Run train_pipeline or POST /api/refresh/{match}.",
                )
            raise HTTPException(
                status_code=503,
                detail=f"Forecast path not available for {match}. Retrain with train_pipeline to enable 14-day path.",
            )
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
            trend_label=str(path_result.get("trend_label", "SIDEWAYS")),
            explanation=str(path_result.get("explanation", "")),
            model_weights=dict(path_result.get("model_weights") or {}),
            volatility_level=str(path_result.get("volatility_level", "UNKNOWN")),
            multi_horizon=multi_horizon,
            mean_path_agreement=float(path_result.get("mean_path_agreement", 0.0)),
            model_agreement_score=float(path_result.get("model_agreement_score", 0.0)),
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
        closes = df["close"].astype(float).round(6).tolist()
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
            features_path = features_dir / f"{match.replace(' ', '_')}_features.parquet"
            if not features_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Feature file not found for {match}. Run train_pipeline or POST /api/refresh/{match}.",
                )
            models_dir = settings.training.models_dir
            coin_models = models_dir / match.replace(" ", "_")
            if not coin_models.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"No trained models found for {match}. Run train_pipeline first.",
                )
            raise HTTPException(
                status_code=503,
                detail=f"Prediction failed for {match}. Check feature file and model files.",
            )
        now = datetime.now(timezone.utc).isoformat()
        return CoinPredictionResponse(
            coin=result["coin"],
            model_predictions=result["predictions"],
            average_prediction=result["average_prediction"],
            lower_bound=result["lower_bound"],
            upper_bound=result["upper_bound"],
            model_agreement_score=result["model_agreement_score"],
            horizon_days=result["horizon_days"],
            generated_at=now,
        )

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
