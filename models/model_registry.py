"""
Model registry: instantiate models by name.
"""

from __future__ import annotations

from typing import Any

from models.base_model import BaseModel
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel

_REGISTRY: dict[str, type[BaseModel]] = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
}

# Fixed seeds + mildly different capacity / sampling → decorrelated errors; same data, different fits.
_DEFAULT_ESTIMATOR_KWARGS: dict[str, dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 16,
        "min_samples_leaf": 2,
        "random_state": 41,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 240,
        "max_depth": 6,
        "learning_rate": 0.046,
        "subsample": 0.78,
        "colsample_bytree": 0.72,
        "reg_lambda": 1.1,
        "random_state": 97,
        "n_jobs": -1,
    },
    "lightgbm": {
        "n_estimators": 280,
        "max_depth": 8,
        "learning_rate": 0.042,
        "subsample": 0.85,
        "colsample_bytree": 0.88,
        "reg_lambda": 1.0,
        "random_state": 123,
        "n_jobs": -1,
    },
}


def get_model(name: str, **kwargs: Any) -> BaseModel:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        One of: random_forest, xgboost, lightgbm.

    Returns
    -------
    BaseModel
        Unfitted model instance.

    Raises
    ------
    KeyError
        If name is not registered.
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")
    merged: dict[str, Any] = dict(_DEFAULT_ESTIMATOR_KWARGS.get(key, {}))
    merged.update(kwargs)
    return _REGISTRY[key](**merged)


def list_models() -> list[str]:
    """Return list of registered model names."""
    return list(_REGISTRY.keys())
