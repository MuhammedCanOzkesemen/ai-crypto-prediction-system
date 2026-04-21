"""
Classify on-disk artifacts as production vs legacy / degraded (before and after inference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from features.schema import FEATURE_SCHEMA_VERSION, official_training_feature_columns
from prediction.inference_utils import load_scaler_bundle_from_disk, load_training_metadata
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def evaluate_artifact_bundle(
    coin_dir: Path,
    slug: str,
    evaluation_dir: Path,
    *,
    live_feature_columns: pd.Index | None = None,
) -> dict[str, Any]:
    """
    Inspect artifacts on disk only (no forward pass).

    Returns artifact_bundle_tier: production | degraded | legacy
    and booleans for API / audit.
    """
    bundle = load_scaler_bundle_from_disk(coin_dir, slug)
    meta = load_training_metadata(coin_dir)
    eval_path = evaluation_dir / f"{slug}_metrics.json"
    has_eval = eval_path.exists()
    has_scaler = bundle.get("scaler") is not None
    has_fc = bool(bundle.get("feature_columns"))
    has_meta = bool(meta)
    v_meta = meta.get("feature_schema_version")
    try:
        v_art = int(v_meta) if v_meta is not None else None
    except (TypeError, ValueError):
        v_art = None

    expected = official_training_feature_columns()
    saved = list(bundle.get("feature_columns") or [])
    columns_match_official = saved == expected
    schema_version_match = v_art == FEATURE_SCHEMA_VERSION

    live_ok = True
    feature_match_ratio = 1.0
    if live_feature_columns is not None and saved:
        present = sum(1 for c in saved if c in live_feature_columns)
        feature_match_ratio = float(present) / float(len(saved)) if saved else 0.0
        live_ok = feature_match_ratio >= 0.999

    production_disk = (
        has_scaler
        and has_fc
        and has_meta
        and has_eval
        and schema_version_match
        and columns_match_official
    )

    if production_disk and live_ok:
        tier = "production"
    elif production_disk and not live_ok:
        tier = "degraded"
    elif has_scaler or has_fc or has_meta:
        tier = "degraded"
    else:
        tier = "legacy"

    return {
        "artifact_bundle_tier": tier,
        "has_scaler": has_scaler,
        "has_saved_feature_columns": has_fc,
        "has_training_metadata": has_meta,
        "has_evaluation_metrics": has_eval,
        "feature_schema_version_artifact": v_art,
        "feature_schema_version_expected": FEATURE_SCHEMA_VERSION,
        "schema_version_match": schema_version_match,
        "columns_match_official_list": columns_match_official,
        "live_columns_cover_saved_features": live_ok,
        "feature_match_ratio": round(feature_match_ratio, 6),
        "training_completed_at": meta.get("trained_at_utc"),
        "expected_feature_count": len(expected),
        "saved_feature_count": len(saved),
    }


def resolve_forecast_artifact_mode(
    bundle_audit: dict[str, Any],
    inference_diag: dict[str, Any],
) -> str:
    """
    Final user-facing mode after one forecast.

    production — only if disk tier is production and inference did not use emergency paths.
    degraded — partial parity or align / high imputation.
    legacy — missing critical artifacts or schema mismatch on disk.
    """
    tier = bundle_audit.get("artifact_bundle_tier", "legacy")
    if tier == "legacy":
        return "legacy"
    if inference_diag.get("used_emergency_width_align"):
        return "degraded"
    if float(inference_diag.get("max_missing_feature_fill_ratio", 0.0)) > 0.08:
        return "degraded"
    if inference_diag.get("scaler_transform_failed"):
        return "degraded"
    if not bundle_audit.get("schema_version_match"):
        return "legacy"
    if not bundle_audit.get("columns_match_official_list"):
        return "degraded"
    if tier == "degraded":
        return "degraded"
    if inference_diag.get("is_constant_prediction"):
        return "degraded"
    return "production"


def forecast_audit_api_subset(bundle_audit: dict[str, Any]) -> dict[str, Any]:
    """Small JSON-safe dict for API responses (no large nested blobs)."""
    if not bundle_audit:
        return {}
    return {
        "artifact_bundle_tier": bundle_audit.get("artifact_bundle_tier"),
        "has_scaler": bundle_audit.get("has_scaler"),
        "has_saved_feature_columns": bundle_audit.get("has_saved_feature_columns"),
        "has_training_metadata": bundle_audit.get("has_training_metadata"),
        "has_evaluation_metrics": bundle_audit.get("has_evaluation_metrics"),
        "feature_schema_version_artifact": bundle_audit.get("feature_schema_version_artifact"),
        "feature_schema_version_expected": bundle_audit.get("feature_schema_version_expected"),
        "schema_version_match": bundle_audit.get("schema_version_match"),
        "columns_match_official_list": bundle_audit.get("columns_match_official_list"),
        "live_columns_cover_saved_features": bundle_audit.get("live_columns_cover_saved_features"),
        "feature_match_ratio": bundle_audit.get("feature_match_ratio"),
        "training_completed_at": bundle_audit.get("training_completed_at"),
        "saved_feature_count": bundle_audit.get("saved_feature_count"),
        "expected_feature_count": bundle_audit.get("expected_feature_count"),
    }
