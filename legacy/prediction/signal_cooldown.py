"""
Signal cooldown: reduce over-trading after actionable tiers fire.

Persists per-coin until ``until_date`` (UTC date) in artifact cache.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils.config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_COOLDOWN_DAYS = 3


def _cooldown_dir() -> Path:
    return settings.data.artifact_dir / "cache" / "signal_cooldown"


def _path_for_coin(coin: str) -> Path:
    slug = (coin or "unknown").replace(" ", "_")
    return _cooldown_dir() / f"{slug}.json"


def cooldown_active(coin: str | None, *, today: date | None = None) -> tuple[bool, str | None]:
    """Returns (blocked, reason message if blocked)."""
    if not coin or not str(coin).strip():
        return False, None
    p = _path_for_coin(coin)
    if not p.exists():
        return False, None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        until_s = data.get("until")
        if not until_s:
            return False, None
        until = date.fromisoformat(str(until_s)[:10])
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as e:
        logger.debug("Cooldown read failed: %s", e)
        return False, None
    t = today or datetime.now(timezone.utc).date()
    if t <= until:
        return True, f"Signal cooldown active until {until.isoformat()}"
    return False, None


def register_actionable_signal(
    coin: str | None,
    trade_decision: str,
    *,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
) -> None:
    """Call when STRONG_BUY or WEAK_BUY is emitted to start cooldown window."""
    if not coin or not str(coin).strip():
        return
    td = str(trade_decision or "").upper()
    if td not in ("STRONG_BUY", "WEAK_BUY"):
        return
    days = max(1, min(14, int(cooldown_days)))
    until = (datetime.now(timezone.utc).date() + timedelta(days=days))
    try:
        _cooldown_dir().mkdir(parents=True, exist_ok=True)
        _path_for_coin(coin).write_text(
            json.dumps(
                {
                    "coin": coin,
                    "until": until.isoformat(),
                    "from_decision": td,
                    "set_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=0,
            ),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Could not write cooldown file: %s", e)
