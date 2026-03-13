"""
Training pipeline for the cryptocurrency forecasting platform.

1. Data + features: fetch price data, build features, save parquet
2. Model training: load features, split train/test, train RF/XGB/LGB, save models
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from utils.config import settings
from utils.constants import COIN_FETCH_INTERVAL_SEC, get_supported_coins_list
from utils.logging_setup import get_logger, configure_root_logger

from data.price_fetcher import fetch_historical_data
from features.feature_builder import build_features, get_feature_columns, TARGET_PRICE_COLUMNS
from models.model_registry import get_model, list_models
from evaluation.model_evaluator import run_evaluation
from evaluation.backtest import BacktestConfig, run_backtest_all

logger = get_logger(__name__)


def run_data_and_features(
    coins: list[str] | None = None,
    days: int | None = None,
    save_features: bool = True,
    inter_coin_delay_sec: float | None = None,
) -> dict[str, object]:
    """
    Load historical data for each coin, build features, save parquet.

    Returns dict mapping coin -> {"price_df": DataFrame, "feature_df": DataFrame}
    """
    configure_root_logger()
    coins = coins or get_supported_coins_list()
    days = days or settings.training.default_history_days
    inter_coin_delay_sec = inter_coin_delay_sec or COIN_FETCH_INTERVAL_SEC

    settings.training.features_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    succeeded: list[str] = []
    failed: list[str] = []

    for i, coin in enumerate(coins):
        if i > 0 and inter_coin_delay_sec > 0:
            time.sleep(inter_coin_delay_sec)
        logger.info("Processing %s/%s: %s (days=%d)", i + 1, len(coins), coin, days)
        price_df = fetch_historical_data(coin, days=days)
        if price_df is None or price_df.empty:
            logger.warning("No data for %s; skipping.", coin)
            results[coin] = {"price_df": None, "feature_df": None}
            failed.append(coin)
            continue
        feature_df = build_features(price_df, include_targets=True)
        if feature_df.empty:
            logger.warning("No features for %s; skipping.", coin)
            results[coin] = {"price_df": price_df, "feature_df": None}
            failed.append(coin)
            continue
        results[coin] = {"price_df": price_df, "feature_df": feature_df}
        succeeded.append(coin)
        if save_features:
            out_path = settings.training.features_dir / f"{coin.replace(' ', '_')}_features.parquet"
            try:
                feature_df.to_parquet(out_path, index=False)
                logger.info("Saved features for %s to %s", coin, out_path)
            except Exception as e:
                logger.exception("Failed to save features for %s: %s", coin, e)
                failed.append(coin)

    n, n_ok, n_fail = len(coins), len(succeeded), len(failed)
    logger.info("Pipeline complete: %d/%d coins succeeded, %d failed. Failed: %s", n_ok, n, n_fail, failed or "none")
    return results


def _get_X_y(feature_df):
    """Extract X and y (14-day multi-output) from feature DataFrame."""
    import numpy as np

    feat_cols = get_feature_columns(include_targets=False)
    avail = [c for c in feat_cols if c in feature_df.columns]
    target_cols = [c for c in TARGET_PRICE_COLUMNS if c in feature_df.columns]
    if not avail or len(target_cols) != 14:
        return None, None
    X = feature_df[avail].astype(np.float64)
    y = feature_df[target_cols].astype(np.float64)  # (n, 14)
    X = X.dropna(axis=0, how="any")
    if X.empty:
        return None, None
    valid_idx = X.index.intersection(y.dropna(how="any").index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    return X.values, y.values  # y shape: (n, 14)


def run_train_models(
    coins: list[str] | None = None,
    train_ratio: float | None = None,
    model_names: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Load feature parquet, split train/test, train models, save to artifacts/models/.

    Returns dict mapping coin -> list of trained model names.
    """
    import numpy as np
    import pandas as pd

    configure_root_logger()
    coins = coins or get_supported_coins_list()
    train_ratio = train_ratio if train_ratio is not None else settings.training.train_ratio
    model_names = model_names or list_models()

    settings.training.models_dir.mkdir(parents=True, exist_ok=True)
    trained: dict[str, list[str]] = {}

    for coin in coins:
        path = settings.training.features_dir / f"{coin.replace(' ', '_')}_features.parquet"
        if not path.exists():
            logger.warning("No features for %s at %s; skipping.", coin, path)
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            logger.exception("Failed to load %s: %s", path, e)
            continue
        X, y = _get_X_y(df)
        if X is None or y is None or len(X) < 50:
            logger.warning("Insufficient data for %s (rows=%s); skipping.", coin, len(X) if X is not None else 0)
            continue
        n = len(X)
        split = int(n * train_ratio)
        if split < 30:
            logger.warning("Train set too small for %s; skipping.", coin)
            continue
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        coin_dir = settings.training.models_dir / coin.replace(" ", "_")
        coin_dir.mkdir(parents=True, exist_ok=True)
        trained[coin] = []
        for name in model_names:
            try:
                model = get_model(name)
                model.fit(X_train, y_train)
                save_path = coin_dir / f"{name}.joblib"
                model.save(save_path)
                trained[coin].append(name)
                logger.info("Trained %s for %s", name, coin)
            except Exception as e:
                logger.exception("Failed to train %s for %s: %s", name, coin, e)
    return trained


def run_full_evaluation(
    coins: list[str] | None = None,
    evaluation_dir: Path | None = None,
) -> dict:
    """
    Run model evaluation and backtest, save to artifacts/evaluation/.

    Returns dict with evaluation and backtest results.
    """
    eval_dir = Path(evaluation_dir) if evaluation_dir else settings.training.features_dir.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    coins = coins or get_supported_coins_list()

    logger.info("Step 3a: Model evaluation (test split metrics)")
    eval_results = run_evaluation(coins=coins, evaluation_dir=eval_dir)
    logger.info("Step 3b: Walk-forward backtest")
    bt_results = run_backtest_all(coins=coins, evaluation_dir=eval_dir, config=BacktestConfig())
    logger.info("Evaluation complete. Results in %s", eval_dir)
    return {"evaluation": eval_results, "backtest": bt_results}


def main() -> None:
    """CLI entry: run data + features, train models, then evaluation."""
    configure_root_logger()
    logger.info("Step 1: Data + features")
    run_data_and_features()
    logger.info("Step 2: Train models")
    run_train_models()
    logger.info("Step 3: Evaluation + backtest")
    run_full_evaluation()
    logger.info("Training pipeline finished.")


if __name__ == "__main__":
    main()
