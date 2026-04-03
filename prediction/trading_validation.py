"""
Production validation: path stability, volatility shock, trend confirmation, chaotic disagreement.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_path_stability_score(path: list[dict[str, Any]] | None) -> float:
    """
    [0, 1]: high when daily path returns are smooth with few direction flips.
    Uses predicted_log_return along the horizon.
    """
    if not path or len(path) < 2:
        return 0.55
    rs = [float(p.get("predicted_log_return") or 0.0) for p in path]
    arr = np.asarray(rs, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return 0.35
    var = float(np.var(arr))
    # Normalize variance: typical daily lr variance often < 1e-4 to few e-4
    var_n = float(max(0.0, min(1.0, 1.0 - min(1.0, var / 2.5e-4))))
    signs = np.sign(arr)
    flips = int(np.sum(signs[1:] * signs[:-1] < 0))
    max_flips = max(len(arr) - 1, 1)
    flip_pen = 1.0 - 0.55 * (flips / max_flips)
    out = 0.52 * var_n + 0.48 * max(0.0, min(1.0, flip_pen))
    return float(max(0.0, min(1.0, round(out, 4))))


def detect_volatility_shock(
    current_vol: float | None,
    historical_vols: list[float] | None,
    *,
    ratio_threshold: float = 1.8,
    min_samples: int = 8,
) -> tuple[bool, float]:
    """
    Shock if current EWMA vol > ratio_threshold × mean of recent history.

    Returns (shock_detected, ratio_or_1.0).
    """
    try:
        cv = float(current_vol) if current_vol is not None else 0.0
    except (TypeError, ValueError):
        cv = 0.0
    if not historical_vols or len(historical_vols) < min_samples:
        return False, 1.0
    hist = [float(x) for x in historical_vols if x is not None and math.isfinite(float(x)) and float(x) > 0]
    if len(hist) < min_samples:
        return False, 1.0
    m = float(np.mean(hist))
    if m < 1e-12:
        return False, 1.0
    r = cv / m
    return bool(r > ratio_threshold), float(round(r, 4))


def compute_trend_confirmation_score(
    ema_short: float | None,
    ema_long: float | None,
    adx_last5: list[float] | None,
    *,
    forecast_bullish: bool,
) -> float:
    """
    [0, 1]: EMA20 > EMA50 supports long; ADX persistence (not collapsing) adds weight.
    """
    try:
        es = float(ema_short) if ema_short is not None else 0.0
        el = float(ema_long) if ema_long is not None else 0.0
    except (TypeError, ValueError):
        es, el = 0.0, 0.0
    if es <= 0 or el <= 0:
        return 0.45
    ema_score = 1.0 if es > el else 0.35 if es < el else 0.55
    if not forecast_bullish:
        ema_score *= 0.82
    adxs = [float(x) for x in (adx_last5 or []) if x is not None and math.isfinite(float(x))]
    if len(adxs) < 3:
        persist = 0.5
    else:
        mean_adx = float(np.mean(adxs))
        std_adx = float(np.std(adxs))
        persist = float(max(0.0, min(1.0, 1.0 - min(1.0, std_adx / max(mean_adx, 8.0)))))
    out = 0.55 * ema_score + 0.45 * persist
    return float(max(0.0, min(1.0, round(out, 4))))


def lr_matrix_chaotic_disagreement(lr_matrix: list[list[float]] | None) -> tuple[bool, float]:
    """
    Chaotic = many steps where models strongly disagree in *sign* (not just magnitude).

    Returns (is_chaotic, chaos_intensity in [0,1]).
    """
    if not lr_matrix:
        return False, 0.0
    chaotic_steps = 0
    n = 0
    for row in lr_matrix:
        vals = [float(x) for x in row if math.isfinite(float(x))]
        if len(vals) < 2:
            continue
        n += 1
        s = np.sign(np.asarray(vals, dtype=np.float64))
        pos = float(np.mean(s > 0))
        neg = float(np.mean(s < 0))
        if pos >= 0.34 and neg >= 0.34:
            chaotic_steps += 1
    if n == 0:
        return False, 0.0
    frac = chaotic_steps / float(n)
    chaotic = frac >= 0.28
    return chaotic, float(round(frac, 4))


def compose_risk_adjusted_confidence(
    base_confidence: float,
    mean_path_agreement: float,
    stability_score: float,
    consensus_score: float,
    trend_confirmation_score: float,
    *,
    volatility_regime_high: bool,
    volatility_shock: bool,
    chaotic_disagreement: bool,
    market_regime: str | None = None,
) -> float:
    """
    confidence ≈ base × agreement_factor × stability × consensus × trend_confirm;
    market regime: VOLATILE ×0.75, TRENDING ×1.05, RANGING ×0.96;
    then hard caps on high vol / shock / chaos.
    """
    bc = float(max(0.0, min(1.0, base_confidence)))
    agree = float(max(0.0, min(1.0, mean_path_agreement)))
    stab = float(max(0.25, min(1.0, stability_score)))
    cons = float(max(0.22, min(1.0, consensus_score)))
    tconf = float(max(0.25, min(1.0, trend_confirmation_score)))
    agree_f = 0.48 + 0.52 * agree
    c = bc * agree_f * stab * cons * tconf
    mr = str(market_regime or "").strip().upper()
    if mr == "VOLATILE":
        c *= 0.75
    elif mr == "TRENDING":
        c = min(1.0, c * 1.05)
    elif mr == "RANGING":
        c *= 0.96
    c = float(max(0.0, min(1.0, c)))
    if chaotic_disagreement:
        c = min(c, 0.5)
    if volatility_regime_high or volatility_shock:
        c = min(c, 0.6)
    return float(round(c, 4))
