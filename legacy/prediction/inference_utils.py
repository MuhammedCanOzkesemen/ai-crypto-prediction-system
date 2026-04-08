"""
Shared inference helpers: feature column resolution and row alignment for training vs live data.
"""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.feature_builder import get_feature_columns
from utils.logging_setup import get_logger

logger = get_logger(__name__)

# Canonical bundle name under each coin model directory; aliases tried in order.
SCALER_PRIMARY = "feature_scaler.joblib"


def scaler_artifact_paths(coin_dir: Path, slug: str) -> list[Path]:
    """Paths to try when loading a saved scaler bundle (newest convention first)."""
    return [
        coin_dir / SCALER_PRIMARY,
        coin_dir / f"{slug}_feature_scaler.joblib",
        coin_dir / f"{slug}_scaler.joblib",
        coin_dir / f"{slug}_scaler.pkl",
    ]


def load_scaler_bundle_from_disk(coin_dir: Path, slug: str) -> dict[str, Any]:
    """
    Load scaler + feature_columns if present. Never raises; returns a dict suitable
    for recursive_forecast / single-step predict.

    Keys: scaler (optional), feature_columns (optional), source_path (optional).
    """
    out: dict[str, Any] = {"scaler": None, "feature_columns": None, "source_path": None}
    for p in scaler_artifact_paths(coin_dir, slug):
        if not p.exists():
            continue
        try:
            raw = joblib.load(p)
        except Exception as e:
            logger.warning("Could not read scaler artifact %s: %s", p, e)
            continue
        if isinstance(raw, dict):
            out["scaler"] = raw.get("scaler")
            cols = raw.get("feature_columns")
            if cols is not None:
                out["feature_columns"] = [str(c) for c in cols]
        else:
            # Legacy: file contained only a fitted scaler
            out["scaler"] = raw
        out["source_path"] = str(p)
        return out
    return out


def resolve_inference_columns(
    scaler_bundle: dict[str, Any] | None,
    available_in_row: pd.Index,
) -> list[str]:
    """
    Column order for X matching training when possible.

    If training saved feature_columns, use that exact order (fill missing at row build).
    Otherwise use project feature list intersected with available columns.
    """
    bundle = scaler_bundle or {}
    saved = bundle.get("feature_columns")
    if saved:
        return [str(c) for c in saved]
    base = get_feature_columns(include_targets=False)
    return [c for c in base if c in available_in_row]


def feature_row_to_matrix(last: pd.Series, columns: list[str]) -> np.ndarray:
    """Shape (1, n_features); NaN / missing -> 0.0."""
    row, _ = feature_row_to_matrix_with_stats(last, columns)
    return row


def feature_row_to_matrix_with_stats(
    last: pd.Series,
    columns: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Build X row; count columns that were missing/NaN and filled as 0 (padding effect).
    """
    row = np.zeros((1, len(columns)), dtype=np.float64)
    missing = 0
    for j, c in enumerate(columns):
        if c not in last.index:
            missing += 1
            continue
        v = last[c]
        if pd.isna(v):
            missing += 1
            continue
        try:
            row[0, j] = float(v)
        except (TypeError, ValueError):
            missing += 1
    n = len(columns) if columns else 1
    stats = {
        "missing_feature_fill_ratio": float(missing) / float(n),
        "zero_value_ratio": float(np.mean(row == 0.0)),
    }
    return row, stats


def prediction_matrix_to_dataframe(
    X: np.ndarray,
    *,
    model_wrapper: Any,
    training_feature_columns: list[str] | None,
) -> pd.DataFrame:
    """
    Wrap a numeric feature matrix as a pandas DataFrame for ``model.predict``.

    Prefer the fitted estimator's ``feature_names_in_`` when present and width matches
    (avoids LGBM/XGB warnings). Otherwise use saved training column names in order.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = int(X.shape[1])
    inner = getattr(model_wrapper, "model", model_wrapper)
    fin = getattr(inner, "feature_names_in_", None)
    if fin is not None and len(fin) == n:
        col_list = [str(x) for x in list(fin)]
        out = pd.DataFrame(X, columns=col_list)
        assert list(out.columns) == col_list
        return out

    tfc = [str(c) for c in training_feature_columns] if training_feature_columns else None
    if tfc is not None and len(tfc) == n:
        out = pd.DataFrame(X, columns=tfc)
        assert list(out.columns) == tfc
        return out
    if tfc is not None and len(tfc) > n:
        use = tfc[:n]
        out = pd.DataFrame(X, columns=use)
        return out
    if tfc is not None and len(tfc) < n:
        pad_n = n - len(tfc)
        names = tfc + [f"__inference_pad_{j}" for j in range(pad_n)]
        out = pd.DataFrame(X, columns=names)
        return out
    return pd.DataFrame(X)


def infer_sklearn_n_features_in(model_wrapper: Any) -> int | None:
    """Best-effort feature count from a fitted sklearn / xgboost / lightgbm estimator."""
    inner = getattr(model_wrapper, "model", model_wrapper)
    for attr in ("n_features_in_", "n_features_in", "n_features_"):
        n = getattr(inner, attr, None)
        if n is not None:
            try:
                return int(n)
            except (TypeError, ValueError):
                continue
    return None


def infer_n_features_for_ensemble(models: dict[str, Any]) -> int | None:
    """Single n_features for predict(); warns if models disagree."""
    seen: list[tuple[str, int]] = []
    for name, m in models.items():
        n = infer_sklearn_n_features_in(m)
        if n is not None:
            seen.append((name, n))
    if not seen:
        return None
    counts = {n for _, n in seen}
    if len(counts) > 1:
        logger.warning(
            "Ensemble models report different n_features_in_: %s; using %d from %s",
            seen,
            seen[0][1],
            seen[0][0],
        )
    return seen[0][1]


def align_X_to_n_features(X: np.ndarray, n_expected: int | None, *, allow: bool = True) -> tuple[np.ndarray, bool]:
    """
    Pad (zeros) or truncate columns so X matches estimators trained on a fixed width.

    Returns (X_aligned, emergency_align_used). If allow is False and widths differ, returns (X, False).
    """
    if n_expected is None:
        return X, False
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n = int(X.shape[1])
    if n == n_expected:
        return X, False
    if not allow:
        logger.warning(
            "Feature width mismatch (%d vs %d) and emergency align disabled — predictions may fail",
            n,
            n_expected,
        )
        return X, False
    if n < n_expected:
        logger.warning(
            "Feature width mismatch: got %d columns, model expects %d — padding with zeros (degraded / emergency)",
            n,
            n_expected,
        )
        pad = np.zeros((X.shape[0], n_expected - n), dtype=np.float64)
        return np.hstack([X, pad]), True
    logger.warning(
        "Feature width mismatch: got %d columns, model expects %d — truncating (degraded / emergency)",
        n,
        n_expected,
    )
    return X[:, :n_expected], True


def load_training_metadata(coin_dir: Path) -> dict[str, Any]:
    """Optional JSON written by train_pipeline (historical return stats, schema)."""
    p = coin_dir / "training_metadata.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read training_metadata.json from %s: %s", coin_dir, e)
        return {}


def classify_forecast_quality(
    *,
    has_scaler: bool,
    has_saved_feature_columns: bool,
    emergency_width_align_used: bool,
    scaler_transform_failed_any: bool,
) -> str:
    """production | degraded | legacy-fallback"""
    if not has_scaler:
        return "legacy-fallback"
    if emergency_width_align_used or scaler_transform_failed_any or not has_saved_feature_columns:
        return "degraded"
    return "production"


def exact_training_columns_present(saved_cols: list[str] | None, row_index: pd.Index) -> bool:
    if not saved_cols:
        return False
    return all(c in row_index for c in saved_cols)


def save_scaler_bundle(coin_dir: Path, slug: str, scaler: Any, feature_columns: list[str]) -> None:
    """Persist bundle to canonical path and a slug alias (same contents)."""
    coin_dir.mkdir(parents=True, exist_ok=True)
    payload = {"scaler": scaler, "feature_columns": list(feature_columns)}
    primary = coin_dir / SCALER_PRIMARY
    joblib.dump(payload, primary)
    alias = coin_dir / f"{slug}_scaler.joblib"
    if alias.resolve() != primary.resolve():
        joblib.dump(payload, alias)


def slice_scaled_X_for_model(
    Xs: np.ndarray,
    model_name: str,
    training_meta: dict[str, Any] | None,
    full_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Subset scaled feature row(s) to the columns each ensemble member was trained on.

    Legacy artifacts without ``per_model_feature_indices`` use the full width.
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    if Xs.ndim == 1:
        Xs = Xs.reshape(1, -1)
    n = int(Xs.shape[1])
    meta = training_meta or {}
    raw = meta.get("per_model_feature_indices")
    if not isinstance(raw, dict):
        use_names = [str(full_columns[i]) for i in range(min(n, len(full_columns)))]
        return Xs, use_names
    idx_list = raw.get(model_name)
    if not isinstance(idx_list, list) or not idx_list:
        use_names = [str(full_columns[i]) for i in range(min(n, len(full_columns)))]
        return Xs, use_names
    idx = [int(i) for i in idx_list if 0 <= int(i) < n]
    if not idx:
        use_names = [str(full_columns[i]) for i in range(min(n, len(full_columns)))]
        return Xs, use_names
    names = [str(full_columns[i]) for i in idx if i < len(full_columns)]
    return Xs[:, np.array(idx, dtype=np.int64)], names
