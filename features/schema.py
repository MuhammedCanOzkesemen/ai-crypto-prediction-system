"""
Official production feature schema — single versioned contract for train + infer.

All training metadata and artifact audits MUST reference FEATURE_SCHEMA_VERSION.
Feature column order matches ``get_feature_columns(include_targets=False)`` in feature_builder.
"""

from __future__ import annotations

# Bump when changing feature definitions or column order (retrain all coins).
FEATURE_SCHEMA_VERSION: int = 4

TRAINING_TARGET_COLUMN: str = "target_log_return_1d"
TRAINING_TARGET_DESCRIPTION: str = "Next-day log return log(close_{t+1}/close_t); never raw price."


def official_training_feature_columns() -> list[str]:
    """Ordered feature names used for X at train and infer (no target, no OHLCV/date)."""
    from features.feature_builder import get_feature_columns

    return get_feature_columns(include_targets=False)


def official_schema_documentation() -> dict[str, object]:
    """Human- and machine-readable schema summary for metadata/API."""
    cols = official_training_feature_columns()
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "n_features": len(cols),
        "training_target": TRAINING_TARGET_COLUMN,
        "training_target_description": TRAINING_TARGET_DESCRIPTION,
        "feature_columns": cols,
        "notes": (
            "RSI, MACD, BB, EMA/SMA, ATR, rolling vol; returns 1/3/7/14d; ROC; ADX/+DI/-DI; "
            "log-return momentum 3/7/14d; rolling & EWMA log vol 14/30; vol regime; EMA cross; "
            "interactions (momentum×vol, RSI×ADX, MACD×log-ret accel); log_close lags; daily_return lags."
        ),
    }
