"""
Base model interface for cryptocurrency forecasting.

Defines a reusable contract: fit, predict, save, load.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract base class for regression models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> BaseModel:
        """Train the model on (X, y). Returns self for chaining."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for X."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist model to path. Returns path."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> BaseModel:
        """Load model from path. Returns self."""
        ...
