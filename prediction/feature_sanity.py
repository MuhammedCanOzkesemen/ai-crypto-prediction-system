"""
Pre-flight checks on the latest feature row before trusting predictions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# Columns to audit (numeric); skip date-like / IDs
_SKIP_SUBSTR = ("date", "target_")


def audit_feature_row(
    row: pd.Series,
    feature_columns: list[str] | None,
    *,
    abs_cap: float = 1.0e7,
    nan_forbidden: bool = True,
) -> tuple[bool, list[str]]:
    """
    Returns (ok, issues). ok=False → caller should treat as NO_TRADE / degraded.
    """
    issues: list[str] = []
    cols = feature_columns or [c for c in row.index if str(c) not in ("date",)]
    for c in cols:
        if any(sk in str(c).lower() for sk in _SKIP_SUBSTR):
            continue
        if c not in row.index:
            continue
        v = row[c]
        if nan_forbidden and (v is None or (isinstance(v, (float, np.floating)) and math.isnan(float(v)))):
            issues.append(f"NaN in {c}")
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f):
            issues.append(f"Non-finite in {c}")
        elif abs(f) > abs_cap:
            issues.append(f"Extreme magnitude in {c}")
    ok = len(issues) == 0
    return ok, issues[:12]
