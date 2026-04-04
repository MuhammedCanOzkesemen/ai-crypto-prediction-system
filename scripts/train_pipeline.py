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
from models.feature_subsets import indices_dict_for_metadata, per_model_feature_lists

# XGBoost: train on recent history only so it chases regime shifts; RF/LGBM use full train split.
XGBOOST_RECENT_TRAIN_FRACTION = 0.58
from models.model_registry import get_model, list_models
from prediction.inference_utils import save_scaler_bundle
from evaluation.model_evaluator import run_evaluation
from evaluation.backtest import BacktestConfig, run_backtest_all
from prediction.directional_classifier import (
    BUNDLE_VERSION,
    DIRECTIONAL_LABEL_HORIZON_DAYS,
    DIRECTIONAL_RETURN_THRESHOLD_PCT,
    build_directional_labels_forward_close,
)

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
        feature_df = build_features(price_df, include_targets=True, coin=coin)
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


def _get_X_y_with_close(feature_df):
    """
    Extract X, y, feature order, and aligned current close prices.

    Alignment matches the existing regression target construction:
    - features at row t
    - target_log_return_1d for t -> t+1
    - close at row t for forward-return directional labels

    Rows with missing features, target, or close are dropped so downstream
    regression/classification training stays leakage-safe and index-aligned.
    """
    import numpy as np

    feat_cols = get_feature_columns(include_targets=False)
    avail = [c for c in feat_cols if c in feature_df.columns]
    if not avail or TARGET_LOG_RETURN_1D not in feature_df.columns or "close" not in feature_df.columns:
        return None, None, None, None

    X = feature_df[avail].astype(np.float64).dropna(axis=0, how="any")
    if X.empty:
        return None, None, None, None

    y_series = feature_df[TARGET_LOG_RETURN_1D].astype(np.float64)
    close_series = feature_df["close"].astype(np.float64)

    valid_idx = X.index
    valid_idx = valid_idx.intersection(y_series.dropna(how="any").index)
    valid_idx = valid_idx.intersection(close_series.dropna(how="any").index)
    if len(valid_idx) == 0:
        return None, None, None, None

    X = X.loc[valid_idx]
    y = y_series.loc[valid_idx]
    close = close_series.loc[valid_idx]
    return X.values, y.values, avail, close.values


def _train_and_save_directional_classifier(
    *,
    coin: str,
    coin_dir: Path,
    X_train_s,
    y_train,
    close_train,
) -> None:
    """
    Train and persist the directional classifier bundle.

    This step is mandatory for the hybrid forecasting stack. It raises on any
    invalid input, training failure, or missing artifact after save so the
    pipeline cannot silently finish without directional intelligence artifacts.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    logger.info("Training directional classifier for %s...", coin)

    if close_train is None:
        raise RuntimeError(f"{coin}: missing close alignment for directional classifier training")

    X_train_s = np.asarray(X_train_s, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    close_train = np.asarray(close_train, dtype=np.float64).ravel()

    if X_train_s.ndim != 2 or X_train_s.shape[0] == 0:
        raise RuntimeError(f"{coin}: X_train for directional classifier is empty")
    if y_train.ndim != 1 or y_train.shape[0] == 0:
        raise RuntimeError(f"{coin}: y_train for directional classifier is empty")
    if close_train.shape[0] != X_train_s.shape[0]:
        raise RuntimeError(
            f"{coin}: close/X alignment mismatch for directional classifier "
            f"({close_train.shape[0]} vs {X_train_s.shape[0]})"
        )

    y_cls_full, valid_mask = build_directional_labels_forward_close(
        close_train,
        horizon=DIRECTIONAL_LABEL_HORIZON_DAYS,
        threshold_pct=DIRECTIONAL_RETURN_THRESHOLD_PCT,
    )
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape[0] != X_train_s.shape[0]:
        raise RuntimeError(
            f"{coin}: directional label mask/X mismatch ({valid_mask.shape[0]} vs {X_train_s.shape[0]})"
        )
    if not np.any(valid_mask):
        raise RuntimeError(f"{coin}: directional classifier produced zero valid labels")

    X_dir = X_train_s[valid_mask]
    y_cls = np.asarray(y_cls_full[valid_mask], dtype=np.int64)

    if X_dir.shape[0] == 0:
        raise RuntimeError(f"{coin}: directional classifier has zero usable training rows")
    if len(np.unique(y_cls)) < 2:
        raise RuntimeError(
            f"{coin}: directional classifier requires at least 2 classes, got {sorted(np.unique(y_cls).tolist())}"
        )

    label_counts = {
        "down": int(np.sum(y_cls == 0)),
        "neutral": int(np.sum(y_cls == 1)),
        "up": int(np.sum(y_cls == 2)),
    }
    logger.info(
        "Directional labels for %s: rows=%d down=%d neutral=%d up=%d horizon=%d threshold=%.4f",
        coin,
        int(X_dir.shape[0]),
        label_counts["down"],
        label_counts["neutral"],
        label_counts["up"],
        DIRECTIONAL_LABEL_HORIZON_DAYS,
        DIRECTIONAL_RETURN_THRESHOLD_PCT,
    )

    lr_clf = LogisticRegression(
        max_iter=1200,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    lr_clf.fit(X_dir, y_cls)

    xgb_clf = XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    xgb_clf.fit(X_dir, y_cls)

    save_path = coin_dir / "directional_classifier_bundle.joblib"
    legacy_path = coin_dir / "directional_classifier.joblib"
    logger.info("Saving directional classifier for %s to %s...", coin, save_path)

    bundle = {
        "version": BUNDLE_VERSION,
        "logistic_regression": lr_clf,
        "xgboost_classifier": xgb_clf,
        "horizon_days": DIRECTIONAL_LABEL_HORIZON_DAYS,
        "threshold_pct": DIRECTIONAL_RETURN_THRESHOLD_PCT,
        "classes": ("down", "neutral", "up"),
    }
    joblib.dump(bundle, save_path)
    joblib.dump(lr_clf, legacy_path)

    if not save_path.exists():
        raise RuntimeError(f"{coin}: directional classifier bundle save failed at {save_path}")
    if not legacy_path.exists():
        raise RuntimeError(f"{coin}: legacy directional classifier save failed at {legacy_path}")

    logger.info("Saved directional classifier for %s", coin)


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
        tup = _get_X_y_with_close(df)
        if tup[0] is None:
            X, y, feat_order = _get_X_y(df)
            close_all = None
        else:
            X, y, feat_order, close_all = tup
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
            meta_base = _build_training_metadata(df, feat_order)
            meta_path = coin_dir / "training_metadata.json"
        except Exception as e:
            logger.exception("Failed to save scaler for %s: %s", coin, e)
            continue

        close_train = close_all[:split] if close_all is not None and len(close_all) >= split else None
        _train_and_save_directional_classifier(
            coin=coin,
            coin_dir=coin_dir,
            X_train_s=X_train_s,
            y_train=y_train,
            close_train=close_train,
        )

        pm_cols = per_model_feature_lists(model_names, feat_order)
        per_model_idx = indices_dict_for_metadata(pm_cols, feat_order)

        trained[coin] = []
        for name in model_names:
            try:
                idx = per_model_idx.get(name)
                if not idx:
                    idx = list(range(len(feat_order)))
                n_tr = len(y_train)
                if str(name).lower() == "xgboost":
                    i0 = max(0, int(n_tr * (1.0 - XGBOOST_RECENT_TRAIN_FRACTION)))
                    row_sl = slice(i0, None)
                else:
                    row_sl = slice(None, None)
                X_fit = X_train_s[row_sl, :][:, np.array(idx, dtype=np.int64)]
                y_fit = y_train[row_sl]
                model = get_model(name)
                model.fit(X_fit, y_fit)
                save_path = coin_dir / f"{name}.joblib"
                model.save(save_path)
                trained[coin].append(name)
                logger.info("Trained %s for %s on %d/%d features", name, coin, len(idx), len(feat_order))
            except Exception as e:
                logger.exception("Failed to train %s for %s: %s", name, coin, e)

        try:
            meta = dict(meta_base)
            meta["per_model_feature_indices"] = per_model_idx
            meta["ensemble_scaler_n_features"] = len(feat_order)
            meta["per_model_train_window"] = {
                "random_forest": "full_train_split",
                "lightgbm": "full_train_split",
                "xgboost": f"last_{int(XGBOOST_RECENT_TRAIN_FRACTION * 100)}pct_train_rows",
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info("Saved training_metadata.json for %s (per-model feature subsets)", coin)
        except Exception as e:
            logger.exception("Failed to write training_metadata.json for %s: %s", coin, e)
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
