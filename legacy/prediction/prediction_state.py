"""
Persist last published forecast path per coin for output inertia (stable refreshes).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_INERTIA_ALPHA = 0.58  # weight on new path; higher = more responsive


def _slug(coin: str) -> str:
    return coin.replace(" ", "_")


def state_path(cache_dir: Path, coin: str) -> Path:
    return cache_dir / f"{_slug(coin)}_forecast_state.json"


def load_last_path(cache_dir: Path, coin: str) -> list[float] | None:
    p = state_path(cache_dir, coin)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        prices = data.get("prices")
        if isinstance(prices, list) and len(prices) == 14:
            return [float(x) for x in prices]
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
        logger.debug("No usable forecast state for %s: %s", coin, e)
    return None


def load_last_trend_score(cache_dir: Path, coin: str) -> float | None:
    p = state_path(cache_dir, coin)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("trend_score")
        if ts is not None:
            return float(ts)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def save_state(
    cache_dir: Path,
    coin: str,
    prices: list[float],
    trend_score: float,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = state_path(cache_dir, coin)
    payload = {
        "coin": coin,
        "prices": [round(float(x), 8) for x in prices],
        "trend_score": round(float(trend_score), 6),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to save forecast state for %s: %s", coin, e)


def blend_path_with_previous(
    new_prices: list[float],
    prev_prices: list[float] | None,
    *,
    alpha: float = DEFAULT_INERTIA_ALPHA,
) -> list[float]:
    """Blend new 14-day path with previous publication."""
    if not prev_prices or len(prev_prices) != len(new_prices):
        return list(new_prices)
    a = max(0.35, min(0.85, float(alpha)))
    out = [a * n + (1.0 - a) * o for n, o in zip(new_prices, prev_prices)]
    return out
