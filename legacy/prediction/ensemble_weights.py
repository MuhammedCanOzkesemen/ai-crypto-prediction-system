"""
Per-coin ensemble weights derived from evaluation metrics (inverse MAE / RMSE).

Weights are normalized over models that participate in the forecast; persisted
for audit alongside live metrics-derived computation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logging_setup import get_logger

logger = get_logger(__name__)

EPS = 1e-12


def load_evaluation_metrics(evaluation_dir: Path, coin: str) -> dict[str, dict[str, float]]:
    """Load per-model metrics JSON for a coin. Returns {} if missing."""
    path = evaluation_dir / f"{coin.replace(' ', '_')}_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read metrics for %s: %s", coin, e)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for name, m in data.items():
        if isinstance(m, dict):
            out[str(name)] = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
    return out


def compute_weights_from_metrics(
    model_names: list[str],
    metrics_by_model: dict[str, dict[str, float]],
    *,
    prefer: str = "mae",
) -> dict[str, float]:
    """
    Normalized weights proportional to 1/error. Uses MAE if present and positive,
    else RMSE; else uniform weight for that model.
    """
    if not model_names:
        return {}
    raw: dict[str, float] = {}
    for name in model_names:
        m = metrics_by_model.get(name, {})
        w = 1.0
        if prefer == "mae" and m.get("mae") is not None:
            err = float(m["mae"])
            if err > EPS:
                w = 1.0 / err
            elif m.get("rmse") is not None and float(m["rmse"]) > EPS:
                w = 1.0 / float(m["rmse"])
        elif m.get("rmse") is not None and float(m["rmse"]) > EPS:
            w = 1.0 / float(m["rmse"])
        elif m.get("mae") is not None and float(m["mae"]) > EPS:
            w = 1.0 / float(m["mae"])
        raw[name] = w
    s = sum(raw.values())
    if s < EPS:
        u = 1.0 / len(model_names)
        return {n: u for n in model_names}
    return {n: raw[n] / s for n in model_names}


def persist_ensemble_weights(
    coin: str,
    weights: dict[str, float],
    evaluation_dir: Path,
) -> None:
    """Write `{coin}_ensemble_weights.json` for auditing."""
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    path = evaluation_dir / f"{coin.replace(' ', '_')}_ensemble_weights.json"
    payload: dict[str, Any] = {
        "coin": coin,
        "weights": {k: round(float(v), 6) for k, v in sorted(weights.items())},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved ensemble weights for %s to %s", coin, path)
    except OSError as e:
        logger.warning("Failed to save ensemble weights for %s: %s", coin, e)
