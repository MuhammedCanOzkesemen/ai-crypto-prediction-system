"""
LightGBM regression model for crypto price prediction.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor

from models.base_model import BaseModel


class LightGBMModel(BaseModel):
    """LGBMRegressor wrapper with fit, predict, save, load."""

    def __init__(self, **kwargs):
        self.model = LGBMRegressor(verbose=-1, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> LightGBMModel:
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(np.float64)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load(self, path: str | Path) -> LightGBMModel:
        self.model = joblib.load(path)
        return self
