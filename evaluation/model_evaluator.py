"""
Model evaluator: compute metrics on test split for each trained model.

Metrics: MAE, RMSE, R2, MAPE, directional_accuracy.
Saves per-coin JSON to artifacts/evaluation/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.feature_builder import TARGET_LOG_RETURN_1D, get_feature_columns
from models.model_registry import get_model, list_models
from prediction.inference_utils import (
    load_scaler_bundle_from_disk,
    prediction_matrix_to_dataframe,
    slice_scaled_X_for_model,
)
from prediction.ensemble_weights import compute_weights_from_metrics, persist_ensemble_weights
from utils.config import settings
from utils.constants import get_supported_coins_list
from utils.logging_setup import get_logger, configure_root_logger

logger = get_logger(__name__)

CLOSE_COL = "close"


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """MAPE in [0, 1]; uses eps to avoid div by zero."""
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _directional_accuracy_log_return(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of steps where sign of predicted log return matches realized log return."""
    if len(y_true) == 0:
        return 0.0
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    mask = s_true != 0
    if not np.any(mask):
        return 0.5
    return float(np.mean(s_true[mask] == s_pred[mask]))


def _get_X_y_close(feature_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract X, y (n,) next-day log return, and close from feature DataFrame."""
    feat_cols = get_feature_columns(include_targets=False)
    avail = [c for c in feat_cols if c in feature_df.columns]
    if not avail or TARGET_LOG_RETURN_1D not in feature_df.columns:
        return None, None, None
    X = feature_df[avail].astype(np.float64).dropna(axis=0, how="any")
    if X.empty:
        return None, None, None
    y_series = feature_df[TARGET_LOG_RETURN_1D].astype(np.float64)
    valid_idx = X.index.intersection(y_series.dropna(how="any").index)
    if len(valid_idx) == 0:
        return None, None, None
    y = y_series.loc[valid_idx].values
    close = (
        feature_df.loc[valid_idx, CLOSE_COL].astype(np.float64).values
        if CLOSE_COL in feature_df.columns
        else None
    )
    return X.loc[valid_idx].values, y, close


def evaluate_models(
    coin: str,
    feature_df: pd.DataFrame,
    train_ratio: float,
    model_names: list[str] | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate each trained model on the test split.

    Returns dict: {model_name: {mae, rmse, r2, mape, directional_accuracy}, ...}
    """
    X, y, close = _get_X_y_close(feature_df)
    if X is None or y is None:
        logger.warning("Cannot extract X/y for %s", coin)
        return {}
    if close is None:
        logger.warning("No 'close' column for %s; directional_accuracy set to 0", coin)
        close = y  # fallback: use y as proxy (direction vs current will be wrong)

    n = len(X)
    split = int(n * train_ratio)
    if split < 30 or n - split < 10:
        logger.warning("Insufficient test data for %s (train=%d, test=%d)", coin, split, n - split)
        return {}

    X_test = X[split:]
    y_test = y[split:]
    close_test = close[split:] if close is not None else None

    models_dir = Path(models_dir) if models_dir else settings.training.models_dir
    slug = coin.replace(" ", "_")
    coin_dir = models_dir / slug
    bundle = load_scaler_bundle_from_disk(coin_dir, slug)
    saved_col_order = list(bundle.get("feature_columns") or [])
    meta: dict = {}
    meta_path = coin_dir / "training_metadata.json"
    if meta_path.exists():
        try:
            import json

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    scaler = bundle.get("scaler")
    if scaler is not None:
        try:
            X_test_s = scaler.transform(X_test)
        except Exception as e:
            logger.warning(
                "Scaler transform failed for %s (%s); evaluating on raw features (legacy).",
                coin,
                e,
            )
            X_test_s = X_test
    else:
        logger.warning(
            "No scaler artifact for %s; evaluating on raw features (legacy compatibility).",
            coin,
        )
        X_test_s = X_test

    model_names = model_names or list_models()
    results: dict[str, dict[str, float]] = {}

    for name in model_names:
        path = coin_dir / f"{name}.joblib"
        if not path.exists():
            continue
        try:
            model = get_model(name)
            model.load(path)
            Xv, cnames = slice_scaled_X_for_model(X_test_s, name, meta, saved_col_order)
            X_df = prediction_matrix_to_dataframe(
                Xv, model_wrapper=model, training_feature_columns=cnames
            )
            y_pred = np.ravel(model.predict(X_df))
        except Exception as e:
            logger.exception("Failed to evaluate %s for %s: %s", name, coin, e)
            continue

        results[name] = {
            "mae": _mae(y_test, y_pred),
            "rmse": _rmse(y_test, y_pred),
            "r2": _r2(y_test, y_pred),
            "mape": _mape(y_test, y_pred),
            "directional_accuracy": _directional_accuracy_log_return(y_test, y_pred),
        }
    return results


def run_evaluation(
    coins: list[str] | None = None,
    features_dir: Path | None = None,
    models_dir: Path | None = None,
    evaluation_dir: Path | None = None,
    train_ratio: float | None = None,
    save_json: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate all trained models for each coin and save metrics.

    Returns dict: coin -> {model_name: {mae, rmse, r2, mape, directional_accuracy}}
    """
    import pandas as pd

    configure_root_logger()

    coins = coins or get_supported_coins_list()
    features_dir = Path(features_dir) if features_dir else settings.training.features_dir
    models_dir = Path(models_dir) if models_dir else settings.training.models_dir
    eval_dir = Path(evaluation_dir) if evaluation_dir else features_dir.parent / "evaluation"
    train_ratio = train_ratio if train_ratio is not None else settings.training.train_ratio

    eval_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, Any]] = {}

    for coin in coins:
        path = features_dir / f"{coin.replace(' ', '_')}_features.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.exception("Failed to load %s: %s", path, e)
            continue
        metrics = evaluate_models(coin, df, train_ratio, models_dir=models_dir)
        if not metrics:
            continue
        all_results[coin] = metrics
        if save_json:
            out_path = eval_dir / f"{coin.replace(' ', '_')}_metrics.json"
            try:
                with open(out_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.info("Saved metrics for %s to %s", coin, out_path)
                w = compute_weights_from_metrics(list(metrics.keys()), metrics)
                persist_ensemble_weights(coin, w, eval_dir)
            except OSError as e:
                logger.exception("Failed to save metrics for %s: %s", coin, e)

    if save_json and all_results:
        _save_summary_csv(all_results, eval_dir)
    return all_results


def _save_summary_csv(all_results: dict[str, dict[str, Any]], eval_dir: Path) -> None:
    """Write summary_metrics.csv aggregating metrics across coins and models."""
    rows = []
    for coin, models in all_results.items():
        for model_name, m in models.items():
            rows.append({
                "coin": coin,
                "model": model_name,
                "mae": m.get("mae", 0),
                "rmse": m.get("rmse", 0),
                "r2": m.get("r2", 0),
                "mape": m.get("mape", 0),
                "directional_accuracy": m.get("directional_accuracy", 0),
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = eval_dir / "summary_metrics.csv"
    try:
        df.to_csv(path, index=False)
        logger.info("Saved summary_metrics.csv to %s", path)
    except OSError as e:
        logger.exception("Failed to save summary_metrics.csv: %s", e)
