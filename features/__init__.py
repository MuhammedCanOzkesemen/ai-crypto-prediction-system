"""
Feature module: technical indicators and target creation for ML.

Input: price DataFrame (date, open, high, low, close, volume).
Output: ML-ready DataFrame with indicators and optional 14d targets.
"""

from .feature_builder import (
    build_features,
    get_feature_columns,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_ema,
    add_rolling_volatility,
    add_returns,
    add_volume_change,
    add_targets,
)

__all__ = [
    "build_features",
    "get_feature_columns",
    "add_rsi",
    "add_macd",
    "add_bollinger_bands",
    "add_ema",
    "add_rolling_volatility",
    "add_returns",
    "add_volume_change",
    "add_targets",
]
