"""
List, archive, or delete incomplete / schema-mismatched model bundles.

Production bundles (per evaluate_artifact_bundle): scaler, saved columns, metadata,
evaluation metrics, schema version match, official column list.

Examples::

    python scripts/cleanup_legacy_artifacts.py --dry-run
    python scripts/cleanup_legacy_artifacts.py --archive
    python scripts/cleanup_legacy_artifacts.py --delete-legacy --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from prediction.artifact_audit import evaluate_artifact_bundle
from utils.config import settings
from utils.constants import get_supported_coins_list
from utils.logging_setup import configure_root_logger, get_logger

logger = get_logger(__name__)


def main() -> None:
    configure_root_logger()
    p = argparse.ArgumentParser(description="Cleanup legacy or incomplete artifact directories.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only (default if neither --archive nor --delete-legacy).",
    )
    p.add_argument(
        "--archive",
        action="store_true",
        help="Move non-production bundles to artifacts/archive/<timestamp>/.",
    )
    p.add_argument(
        "--delete-legacy",
        action="store_true",
        help="Permanently remove directories classified as legacy (not degraded/production).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Required with --delete-legacy.",
    )
    p.add_argument(
        "--include-degraded",
        action="store_true",
        help="Also archive/delete degraded-tier bundles (default: legacy only for delete).",
    )
    args = p.parse_args()

    models_dir = settings.training.models_dir
    eval_dir = settings.training.features_dir.parent / "evaluation"
    if not models_dir.is_dir():
        logger.error("Models dir missing: %s", models_dir)
        sys.exit(1)

    if args.delete_legacy and not args.yes:
        logger.error("--delete-legacy requires --yes")
        sys.exit(2)

    dry = args.dry_run or (not args.archive and not args.delete_legacy)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = settings.training.features_dir.parent / "archive" / f"artifacts_{ts}"

    rows: list[dict] = []
    for coin in get_supported_coins_list():
        slug = coin.replace(" ", "_")
        coin_dir = models_dir / slug
        if not coin_dir.is_dir():
            rows.append({"coin": coin, "slug": slug, "tier": "missing", "path": str(coin_dir)})
            continue
        audit = evaluate_artifact_bundle(coin_dir, slug, eval_dir, live_feature_columns=None)
        tier = str(audit.get("artifact_bundle_tier", "legacy"))
        rows.append({"coin": coin, "slug": slug, "tier": tier, "path": str(coin_dir), "audit": audit})

    for r in rows:
        tier = r["tier"]
        if tier == "missing":
            print(f"[missing] {r['coin']}: no directory")
            continue
        audit = r.get("audit") or {}
        print(
            f"[{tier}] {r['coin']}: scaler={audit.get('has_scaler')} "
            f"meta={audit.get('has_training_metadata')} eval={audit.get('has_evaluation_metrics')} "
            f"schema_ok={audit.get('schema_version_match')} cols_ok={audit.get('columns_match_official_list')}"
        )

        if tier == "production":
            continue

        eligible_archive = tier == "legacy" or (args.include_degraded and tier == "degraded")

        if dry and not args.archive and not args.delete_legacy:
            if tier == "legacy":
                print(f"  -> eligible: --archive or --delete-legacy --yes  ({r['path']})")
            elif tier == "degraded":
                print(f"  -> eligible: --archive --include-degraded  ({r['path']})")
            continue

        if args.archive and eligible_archive:
            if dry:
                print(f"  -> would archive {r['path']}")
            else:
                archive_root.mkdir(parents=True, exist_ok=True)
                dest = archive_root / r["slug"]
                if dest.exists():
                    dest = archive_root / f"{r['slug']}_dup"
                shutil.move(r["path"], dest)
                logger.info("Archived %s -> %s", r["path"], dest)
        elif args.delete_legacy and tier == "legacy":
            if dry:
                print(f"  -> would delete {r['path']}")
            else:
                shutil.rmtree(r["path"])
                logger.info("Deleted legacy bundle %s", r["path"])

    summary_path = archive_root / "cleanup_summary.json" if args.archive and not dry else None
    if summary_path:
        summary_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %s", summary_path)

    if dry:
        logger.info("Dry run complete. Use --archive or --delete-legacy --yes to modify disk.")


if __name__ == "__main__":
    main()
