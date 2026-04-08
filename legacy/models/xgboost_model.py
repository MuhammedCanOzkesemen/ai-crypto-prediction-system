"""
XGBoost regression model for crypto price prediction.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRegressor

from models.base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBRegressor wrapper with fit, predict, save, load."""

    def __init__(self, **kwargs):
        self.model = XGBRegressor(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> XGBoostModel:
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float64)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load(self, path: str | Path) -> XGBoostModel:
        self.model = joblib.load(path)
        return self
