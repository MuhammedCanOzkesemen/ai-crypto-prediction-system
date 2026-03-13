"""
Walk-forward backtest for time-series forecasting.

Design:
- No random shuffle; preserve time order.
- Train on [0 : train_end], test on [train_end : train_end + test_size].
- Roll forward: train_end += step_size (or step_size = test_size for non-overlapping).
- Extensible: configurable window sizes, step, and metric callbacks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from features.feature_builder import get_feature_columns, TARGET_PRICE_COLUMNS
from models.model_registry import get_model, list_models
from utils.config import settings
from utils.logging_setup import get_logger, configure_root_logger

logger = get_logger(__name__)

CLOSE_COL = "close"
HORIZON_DAYS = 14


@dataclass
class BacktestConfig:
    """Backtest window configuration."""

    train_ratio: float = 0.6
    test_size: int = 30
    step_size: int | None = None  # None = use test_size (non-overlapping)


@dataclass
class BacktestResult:
    """Single backtest fold result."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    predictions: dict[str, list[float]] = field(default_factory=dict)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot < 1e-12 else float(1 - ss_res / ss_tot)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, close: np.ndarray) -> float:
    pred_dir = (y_pred > close).astype(np.int64)
    true_dir = (y_true > close).astype(np.int64)
    return float(np.mean(pred_dir == true_dir)) if len(y_true) > 0 else 0.0


def _get_X_y_close(feature_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract X, y (n, 14) multi-output, and close."""
    feat_cols = get_feature_columns(include_targets=False)
    avail = [c for c in feat_cols if c in feature_df.columns]
    target_cols = [c for c in TARGET_PRICE_COLUMNS if c in feature_df.columns]
    if not avail or len(target_cols) != HORIZON_DAYS:
        return None, None, None
    X = feature_df[avail].astype(np.float64).dropna(axis=0, how="any")
    if X.empty:
        return None, None, None
    valid_idx = X.index.intersection(feature_df[target_cols].dropna(how="any").index)
    if len(valid_idx) == 0:
        return None, None, None
    y = feature_df.loc[valid_idx, target_cols].astype(np.float64).values
    close = (
        feature_df.loc[valid_idx, CLOSE_COL].astype(np.float64).values
        if CLOSE_COL in feature_df.columns
        else feature_df.loc[valid_idx, target_cols[-1]].astype(np.float64).values  # fallback
    )
    return X.loc[valid_idx].values, y, close


def run_backtest(
    coin: str,
    feature_df: pd.DataFrame,
    config: BacktestConfig | None = None,
    model_names: list[str] | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Walk-forward backtest for a coin.

    For each fold:
    - Train on [0 : train_end]
    - Test on [train_end : train_end + test_size]
    - Advance train_end by step_size

    Returns dict with folds, aggregated metrics, and summary.
    """
    config = config or BacktestConfig()
    step = config.step_size if config.step_size is not None else config.test_size

    X, y, close = _get_X_y_close(feature_df)
    if X is None or y is None or close is None:
        return {"coin": coin, "folds": [], "aggregated": {}, "n_folds": 0}

    n = len(X)
    min_train = int(n * config.train_ratio)
    if min_train < 50 or config.test_size < 5:
        return {"coin": coin, "folds": [], "aggregated": {}, "n_folds": 0}

    model_names = model_names or list_models()

    folds: list[dict[str, Any]] = []
    all_metrics: dict[str, dict[str, list[float]]] = {name: {"mae": [], "rmse": [], "r2": [], "directional_accuracy": []} for name in model_names}

    train_end = min_train
    fold_idx = 0

    while train_end + config.test_size <= n:
        train_start = 0
        test_start = train_end
        test_end = train_end + config.test_size

        X_train, X_test = X[train_start:train_end], X[test_start:test_end]
        y_train = y[train_start:train_end]  # (n_train, 14)
        y_test_full = y[test_start:test_end]  # (n_test, 14)
        y_test_14 = y_test_full[:, HORIZON_DAYS - 1]
        close_test = close[test_start:test_end]

        fold_metrics: dict[str, dict[str, float]] = {}

        for name in model_names:
            try:
                model = get_model(name)
                model.fit(X_train, y_train)
                pred_full = model.predict(X_test)  # (n_test, 14)
                y_pred_14 = pred_full[:, HORIZON_DAYS - 1] if pred_full.ndim > 1 else np.ravel(pred_full)
                fold_metrics[name] = {
                    "mae": _mae(y_test_14, y_pred_14),
                    "rmse": _rmse(y_test_14, y_pred_14),
                    "r2": _r2(y_test_14, y_pred_14),
                    "directional_accuracy": _directional_accuracy(y_test_14, y_pred_14, close_test),
                }
                for k, v in fold_metrics[name].items():
                    all_metrics[name][k].append(v)
            except Exception as e:
                logger.debug("Backtest fold %d %s for %s: %s", fold_idx, name, coin, e)

        folds.append({
            "fold": fold_idx,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "metrics": fold_metrics,
        })
        train_end += step
        fold_idx += 1

    aggregated: dict[str, dict[str, float]] = {}
    for name in model_names:
        m = all_metrics.get(name, {})
        aggregated[name] = {}
        for metric, vals in m.items():
            if vals:
                aggregated[name][f"{metric}_mean"] = float(np.mean(vals))
                aggregated[name][f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    return {
        "coin": coin,
        "n_folds": len(folds),
        "config": {
            "train_ratio": config.train_ratio,
            "test_size": config.test_size,
            "step_size": step,
        },
        "folds": folds,
        "aggregated": aggregated,
    }


def run_backtest_all(
    coins: list[str] | None = None,
    features_dir: Path | None = None,
    evaluation_dir: Path | None = None,
    config: BacktestConfig | None = None,
    save_json: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Run backtest for each coin and save results.

    Returns dict: coin -> backtest result dict.
    """
    from utils.constants import get_supported_coins_list

    configure_root_logger()
    coins = coins or get_supported_coins_list()
    features_dir = Path(features_dir) if features_dir else settings.training.features_dir
    eval_dir = Path(evaluation_dir) if evaluation_dir else features_dir.parent / "evaluation"
    config = config or BacktestConfig()

    eval_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}

    for coin in coins:
        path = features_dir / f"{coin.replace(' ', '_')}_features.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.exception("Failed to load %s: %s", path, e)
            continue
        bt = run_backtest(coin, df, config=config)
        results[coin] = bt
        if save_json:
            out_path = eval_dir / f"{coin.replace(' ', '_')}_backtest.json"
            try:
                with open(out_path, "w") as f:
                    json.dump(bt, f, indent=2)
                logger.info("Saved backtest for %s to %s", coin, out_path)
            except OSError as e:
                logger.exception("Failed to save backtest for %s: %s", coin, e)
    return results
