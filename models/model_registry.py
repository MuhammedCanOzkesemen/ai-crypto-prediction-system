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
    return _REGISTRY[key](**kwargs)


def list_models() -> list[str]:
    """Return list of registered model names."""
    return list(_REGISTRY.keys())
