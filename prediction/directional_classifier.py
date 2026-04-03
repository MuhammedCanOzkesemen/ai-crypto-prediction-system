"""
Optional multinomial direction head (down / neutral / up) trained on scaled features.

Complements log-return regression; used at inference for confidence and high-conviction logic.
"""

from __future__ import annotations

import os

import numpy as np

# ~0.08% daily log move band treated as neutral (configurable)
NEUTRAL_EPS_LOGRET: float = float(os.environ.get("DIRECTIONAL_NEUTRAL_EPS_LOGRET", "0.0008"))

CLASS_NAMES: tuple[str, ...] = ("down", "neutral", "up")


def build_directional_labels(y_logret: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Map next-day log returns to 0=down, 1=neutral, 2=up."""
    e = float(NEUTRAL_EPS_LOGRET if eps is None else eps)
    y = np.asarray(y_logret, dtype=np.float64).ravel()
    return np.where(y > e, 2, np.where(y < -e, 0, 1)).astype(np.int64)
