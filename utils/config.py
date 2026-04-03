"""
Central configuration for the cryptocurrency forecasting platform.

Production-oriented, extendable settings structure.
Uses environment variables where appropriate; no secrets in code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Base paths (project root = parent of utils/)
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Settings containers (extend as needed)
# -----------------------------------------------------------------------------

class APISettings:
    """FastAPI and HTTP-related settings."""
    title: str = "Crypto Forecasting API"
    version: str = "0.1.0"
    debug: bool = os.environ.get("API_DEBUG", "0").lower() in ("1", "true", "yes")
    host: str = os.environ.get("API_HOST", "0.0.0.0")
    port: int = int(os.environ.get("API_PORT", "8000"))
    # CORS origins (comma-separated in env; default allow all)
    _cors: list[str] = [x.strip() for x in os.environ.get("CORS_ORIGINS", "*").strip().split(",") if x.strip()]
    cors_origins: list[str] = _cors if _cors else ["*"]


class DataSettings:
    """Data ingestion and storage."""
    default_history_days: int = int(os.environ.get("DATA_DEFAULT_DAYS", "1500"))
    request_timeout_sec: int = int(os.environ.get("DATA_REQUEST_TIMEOUT", "30"))
    request_delay_sec: float = float(os.environ.get("DATA_REQUEST_DELAY", "1.5"))
    max_retries_429: int = int(os.environ.get("DATA_MAX_RETRIES_429", "2"))
    # Data directory under project (raw CSVs, etc.)
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    # Placeholder: future artifact store
    artifact_dir: Path = PROJECT_ROOT / "artifacts"


class TrainingSettings:
    """Training pipeline and model artifacts."""
    default_history_days: int = int(os.environ.get("TRAIN_HISTORY_DAYS", "1500"))
    features_dir: Path = PROJECT_ROOT / "artifacts" / "features"
    models_dir: Path = PROJECT_ROOT / "artifacts" / "models"
    # Placeholder: MLflow
    mlflow_tracking_uri: str | None = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = os.environ.get("MLFLOW_EXPERIMENT", "crypto_forecast")
    # Train/val/test split (fractions)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class DatabaseSettings:
    """PostgreSQL-ready structure (placeholders)."""
    enabled: bool = os.environ.get("DB_ENABLED", "0").lower() in ("1", "true", "yes")
    host: str = os.environ.get("DB_HOST", "localhost")
    port: int = int(os.environ.get("DB_PORT", "5432"))
    name: str = os.environ.get("DB_NAME", "crypto_forecast")
    user: str = os.environ.get("DB_USER", "")
    password: str = os.environ.get("DB_PASSWORD", "")
    # Connection URL built when enabled
    @property
    def url(self) -> str:
        if not self.password:
            return f"postgresql://{self.user}@{self.host}:{self.port}/{self.name}"
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class CacheSettings:
    """Redis-ready structure (placeholders)."""
    enabled: bool = os.environ.get("CACHE_ENABLED", "0").lower() in ("1", "true", "yes")
    host: str = os.environ.get("REDIS_HOST", "localhost")
    port: int = int(os.environ.get("REDIS_PORT", "6379"))
    db: int = int(os.environ.get("REDIS_DB", "0"))
    default_ttl_sec: int = int(os.environ.get("CACHE_TTL_SEC", "3600"))
    # URL for redis client
    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class ModelRegistrySettings:
    """MLflow / model registry placeholders."""
    enabled: bool = bool(os.environ.get("MLFLOW_TRACKING_URI"))
    tracking_uri: str | None = os.environ.get("MLFLOW_TRACKING_URI")
    experiment_name: str = os.environ.get("MLFLOW_EXPERIMENT", "crypto_forecast")


class PredictionSettings:
    """
    Inference-time behavior. Emergency feature width align helps legacy artifacts
    but marks forecasts as degraded; disable for strict production-only parity.
    """
    allow_emergency_feature_align: bool = os.environ.get(
        "ALLOW_EMERGENCY_FEATURE_ALIGN", "0"
    ).lower() in ("1", "true", "yes")


# -----------------------------------------------------------------------------
# Aggregated config (single import point)
# -----------------------------------------------------------------------------

class Settings:
    """Central settings aggregate."""
    project_root: Path = PROJECT_ROOT
    api: APISettings = APISettings()
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()
    prediction: PredictionSettings = PredictionSettings()
    database: DatabaseSettings = DatabaseSettings()
    cache: CacheSettings = CacheSettings()
    model_registry: ModelRegistrySettings = ModelRegistrySettings()


# Singleton for application use
settings = Settings()
