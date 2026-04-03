"""
Training pipeline for the cryptocurrency forecasting platform.

1. Data + features: fetch price data, build features, save parquet
2. Model training: load features, split train/test, train RF/XGB/LGB, save models + scaler
3. Evaluation + walk-forward backtest

CLI examples::

    python scripts/train_pipeline.py
    python scripts/train_pipeline.py --all
    python scripts/train_pipeline.py --coin Bitcoin
    python scripts/train_pipeline.py --coin Bitcoin --skip-data --train-only
    python scripts/train_pipeline.py --eval-only --coin Bitcoin
    python scripts/train_pipeline.py --report artifacts/training_report.json

Artifacts per coin under ``artifacts/models/{Coin}/``:
``feature_scaler.joblib``, ``{Coin}_scaler.joblib``, ``training_metadata.json`` (schema + historical return tails),
and ``*.joblib`` model files.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib

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

from sklearn.preprocessing import RobustScaler

from data.price_fetcher import fetch_historical_data
from features.feature_builder import (
    TARGET_LOG_RETURN_1D,
    build_features,
    get_feature_columns,
)
from features.schema import FEATURE_SCHEMA_VERSION, official_schema_documentation
from models.model_registry import get_model, list_models
from prediction.inference_utils import save_scaler_bundle
from evaluation.model_evaluator import run_evaluation
from evaluation.backtest import BacktestConfig, run_backtest_all
from prediction.directional_classifier import build_directional_labels

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


def _build_training_metadata(feature_df, feat_order: list[str]) -> dict:
    """Persist schema + historical return tails for inference realism checks."""
    import numpy as np

    doc = official_schema_documentation()
    meta: dict = {
        "training_target": TARGET_LOG_RETURN_1D,
        "feature_columns": list(feat_order),
        "n_features": len(feat_order),
        "horizon_days_recursive_inference": 14,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_schema_documentation": doc,
    }
    if "log_return_1d" in feature_df.columns:
        lr = (
            feature_df["log_return_1d"]
            .astype(np.float64)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(lr) > 80:
            meta["historical_abs_logret_p99"] = float(np.percentile(np.abs(lr.values), 99))
        win = (
            feature_df["log_return_1d"]
            .astype(np.float64)
            .replace([np.inf, -np.inf], np.nan)
            .rolling(14, min_periods=7)
            .sum()
            .abs()
            .dropna()
        )
        if len(win) > 40:
            meta["historical_abs_cum_logret_14d_p99"] = float(np.percentile(win.values, 99))
    return meta


def _get_X_y(feature_df):
    """Extract X and y (next-day log return) from feature DataFrame."""
    import numpy as np

    feat_cols = get_feature_columns(include_targets=False)
    avail = [c for c in feat_cols if c in feature_df.columns]
    if not avail or TARGET_LOG_RETURN_1D not in feature_df.columns:
        return None, None, None
    X = feature_df[avail].astype(np.float64)
    y = feature_df[TARGET_LOG_RETURN_1D].astype(np.float64)
    X = X.dropna(axis=0, how="any")
    if X.empty:
        return None, None, None
    valid_idx = X.index.intersection(y.dropna(how="any").index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    return X.values, y.values, avail  # avail = column order for scaler


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
        X, y, feat_order = _get_X_y(df)
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
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        coin_dir = settings.training.models_dir / coin.replace(" ", "_")
        slug = coin.replace(" ", "_")
        try:
            save_scaler_bundle(coin_dir, slug, scaler, feat_order)
            logger.info(
                "Saved feature scaler for %s (%d columns) to %s and %s_scaler.joblib",
                coin,
                len(feat_order),
                coin_dir / "feature_scaler.joblib",
                slug,
            )
            meta = _build_training_metadata(df, feat_order)
            meta_path = coin_dir / "training_metadata.json"
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info("Saved training_metadata.json for %s", coin)
        except Exception as e:
            logger.exception("Failed to save scaler for %s: %s", coin, e)
            continue

        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression

            y_cls = build_directional_labels(y_train)
            if len(np.unique(y_cls)) >= 2:
                dclf = LogisticRegression(
                    max_iter=800,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                    multi_class="auto",
                )
                dclf.fit(X_train_s, y_cls)
                joblib.dump(dclf, coin_dir / "directional_classifier.joblib")
                logger.info("Saved directional_classifier.joblib for %s", coin)
        except Exception as e:
            logger.warning("Directional classifier not trained for %s: %s", coin, e)

        trained[coin] = []
        for name in model_names:
            try:
                model = get_model(name)
                model.fit(X_train_s, y_train)
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


def write_training_summary_report(
    path: Path,
    *,
    coins_run: list[str],
    trained: dict[str, list[str]],
    evaluation_dir: Path,
) -> None:
    """Write JSON summary: schema version, per-coin artifacts and evaluation presence."""
    rows: list[dict] = []
    for coin in coins_run:
        slug = coin.replace(" ", "_")
        mdir = settings.training.models_dir / slug
        feat = settings.training.features_dir / f"{slug}_features.parquet"
        eval_p = evaluation_dir / f"{slug}_metrics.json"
        meta_path = mdir / "training_metadata.json"
        meta_schema = None
        if meta_path.exists():
            try:
                meta_schema = json.loads(meta_path.read_text(encoding="utf-8")).get("feature_schema_version")
            except Exception:
                meta_schema = None
        rows.append({
            "coin": coin,
            "features_parquet_exists": feat.exists(),
            "models_trained": trained.get(coin, []),
            "artifact_dir": str(mdir),
            "training_metadata_exists": meta_path.exists(),
            "feature_schema_version_in_metadata": meta_schema,
            "feature_scaler_exists": (mdir / "feature_scaler.joblib").exists(),
            "evaluation_metrics_exists": eval_p.exists(),
        })
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version_expected": FEATURE_SCHEMA_VERSION,
        "coins": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote training summary report to %s", path)


def main() -> None:
    """CLI entry: optional --coin / --skip-data / --eval-only style flags."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Crypto forecasting: fetch data, build features, train models, evaluate.",
    )
    parser.add_argument(
        "--coin",
        type=str,
        default=None,
        metavar="NAME",
        help="Process only this coin (e.g. Bitcoin). Default: all supported coins.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip Step 1 (fetch OHLCV + feature parquet). Use existing parquets.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run Step 2 only (train models + save scaler). No fetch, no evaluation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run Step 3 only (evaluation + backtest). Expects models and features present.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="PATH",
        help="After a full run (not --eval-only), write JSON training summary to PATH.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Retrain every supported coin (same as omitting --coin; cannot combine with --coin).",
    )
    args = parser.parse_args()
    if args.all and args.coin:
        parser.error("Use either --coin NAME or --all, not both.")
    coins: list[str] | None = None if args.all else ([args.coin] if args.coin else None)

    configure_root_logger()

    if args.eval_only:
        logger.info("Evaluation + backtest only")
        run_full_evaluation(coins=coins)
        logger.info("Done.")
        return

    coins_run = coins or get_supported_coins_list()

    if not args.skip_data:
        logger.info("Step 1: Data + features")
        run_data_and_features(coins=coins)
    else:
        logger.info("Step 1 skipped (--skip-data)")

    logger.info("Step 2: Train models")
    trained = run_train_models(coins=coins)

    if args.train_only:
        logger.info("Step 3 skipped (--train-only). Verify artifacts:")
        for c in coins_run:
            d = settings.training.models_dir / c.replace(" ", "_")
            scaler_ok = (d / "feature_scaler.joblib").exists() or (d / f"{c.replace(' ', '_')}_scaler.joblib").exists()
            logger.info("  %s: models_dir=%s scaler_ok=%s", c, d, scaler_ok)
        if args.report:
            eval_dir = settings.training.features_dir.parent / "evaluation"
            write_training_summary_report(
                Path(args.report),
                coins_run=coins_run,
                trained=trained,
                evaluation_dir=eval_dir,
            )
        logger.info("Train-only finished.")
        return

    logger.info("Step 3: Evaluation + backtest")
    eval_dir = settings.training.features_dir.parent / "evaluation"
    run_full_evaluation(coins=coins, evaluation_dir=eval_dir)
    if args.report:
        write_training_summary_report(
            Path(args.report),
            coins_run=coins_run,
            trained=trained,
            evaluation_dir=eval_dir,
        )
    logger.info("Training pipeline finished.")


if __name__ == "__main__":
    main()
