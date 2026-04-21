"""Prediction module: predictor service for ensemble forecasts."""

from __future__ import annotations

from typing import Any

from .ensemble_weights import compute_weights_from_metrics, load_evaluation_metrics

__all__ = [
    "Predictor",
    "HORIZON_DAYS",
    "compute_weights_from_metrics",
    "load_evaluation_metrics",
]


def __getattr__(name: str) -> Any:
    if name == "Predictor":
        from .predictor import Predictor

        return Predictor
    if name == "HORIZON_DAYS":
        from .predictor import HORIZON_DAYS

        return HORIZON_DAYS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
