"""
Multi-day ensemble direction consensus from single-step log-return predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def consensus_score_from_snapshots(snapshots: list[dict[str, Any]] | None) -> tuple[float, list[str]]:
    """
    snapshots: each dict with keys direction (-1,0,1), optional confidence (0..1).

    Returns (score 0..1, human reasons).
    """
    reasons: list[str] = []
    if not snapshots:
        return 0.45, ["Insufficient history for multi-day consensus"]
    dirs = [int(s.get("direction") or 0) for s in snapshots]
    confs = [float(s.get("confidence") or 0.5) for s in snapshots]
    n = len(dirs)
    if n == 0:
        return 0.45, reasons
    pos = sum(1 for d in dirs if d > 0)
    neg = sum(1 for d in dirs if d < 0)
    flat = sum(1 for d in dirs if d == 0)
    if pos == n:
        score = 0.92
        reasons.append("Last %d days: ensemble direction consistently up" % n)
    elif neg == n:
        score = 0.28
        reasons.append("Last %d days: ensemble direction consistently down (long bias off)" % n)
    elif pos >= max(neg, flat) and pos >= n - 1:
        score = 0.72
        reasons.append("Majority up over recent days (%d/%d)" % (pos, n))
    elif flat == n:
        score = 0.38
        reasons.append("Recent ensemble directions flat / unclear")
    else:
        score = 0.32
        reasons.append("Heavy penalty: recent ensemble directions inconsistent")
    conf_mean = float(np.mean(confs)) if confs else 0.5
    score = float(max(0.0, min(1.0, score * (0.55 + 0.45 * conf_mean))))
    return round(score, 4), reasons
