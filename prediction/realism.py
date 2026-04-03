"""
Risk-control guardrails for recursive forecasts (not cosmetic smoothing).

Caps daily and cumulative log returns using volatility and optional training-time
historical percentiles so short-horizon paths cannot compound into absurd prices.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from utils.logging_setup import get_logger

logger = get_logger(__name__)

# Outer legal bound on any single-day log return (below legacy ±0.15)
OUTER_DAILY_ABS_CAP = 0.12
# Hard ceiling on vol-scaled daily cap (fraction)
HARD_DAILY_CAP = 0.055
# Multiplier × EWMA vol for per-day cap
VOL_SIGMA_DAILY = 3.0
# Cumulative log-return cap: K * vol0 * sqrt(horizon)
CUMULATIVE_VOL_SIGMA = 3.5


def daily_log_return_cap(
    ewma_vol: float,
    *,
    historical_abs_logret_p99: float | None = None,
) -> float:
    """Max |log return| for one step from volatility (+ optional training tail)."""
    v = max(float(ewma_vol), 0.004)
    cap = min(HARD_DAILY_CAP, VOL_SIGMA_DAILY * v)
    if historical_abs_logret_p99 is not None and historical_abs_logret_p99 > 1e-8:
        cap = min(cap, 1.15 * float(historical_abs_logret_p99))
    return float(max(cap, 0.002))


def clip_daily_log_return(
    r_raw: float,
    daily_cap: float,
    outer_cap: float = OUTER_DAILY_ABS_CAP,
) -> float:
    r = float(r_raw)
    r = max(-outer_cap, min(outer_cap, r))
    return max(-daily_cap, min(daily_cap, r))


def cumulative_scale_returns(
    returns: list[float],
    vol0: float,
    horizon: int,
    *,
    historical_abs_cum14_p99: float | None = None,
) -> tuple[list[float], bool]:
    """
    If sum(|returns|) path is too extreme in cumulative log space, scale all
    ensemble returns by a single factor (preserves direction, limits runaway compounding).
    """
    if not returns:
        return returns, False
    vol0 = max(float(vol0), 0.005)
    cap = CUMULATIVE_VOL_SIGMA * vol0 * math.sqrt(float(horizon))
    if historical_abs_cum14_p99 is not None and historical_abs_cum14_p99 > 1e-8:
        cap = min(cap, 1.2 * float(historical_abs_cum14_p99))
    s = float(sum(returns))
    if abs(s) <= cap:
        return returns, False
    factor = cap / abs(s)
    scaled = [float(r) * factor for r in returns]
    logger.warning(
        "Realism guardrail: scaled cumulative log-return (|sum| %.4f → cap %.4f)",
        abs(s),
        cap,
    )
    return scaled, True


def rebuild_path_from_ensemble_returns(
    path: list[dict[str, Any]],
    spot_start: float,
    old_returns: list[float],
    new_returns: list[float],
    vol_k: float = 2.0,
) -> None:
    """Mutate path: scale per-model returns by old→new ensemble ratio each step; rebuild prices."""
    spot = float(spot_start)
    names = list((path[0].get("model_log_returns") or {}).keys()) if path else []
    cum_m = {n: float(spot_start) for n in names}

    for i, row in enumerate(path):
        r_old = float(old_returns[i])
        r_new = float(new_returns[i])
        f_scale = (r_new / r_old) if abs(r_old) > 1e-12 else 1.0
        mlr = dict(row.get("model_log_returns") or {})
        row["model_log_returns"] = {k: round(float(v) * f_scale, 8) for k, v in mlr.items()}

        spot = spot * math.exp(r_new)
        row["predicted_log_return"] = round(r_new, 8)
        row["predicted_simple_return"] = round(math.exp(r_new) - 1.0, 8)
        row["predicted_price"] = round(spot, 8)
        row["ensemble_prediction"] = round(spot, 8)
        vol = float(row.get("ewma_vol_logret", 0.02) or 0.02)
        vol = max(vol, 0.005)
        hw_log = vol_k * vol * math.sqrt(float(i + 1))
        row["lower_bound"] = round(spot * math.exp(-hw_log), 8)
        row["upper_bound"] = round(spot * math.exp(hw_log), 8)

        for n in names:
            lr = float(row["model_log_returns"].get(n, 0.0))
            cum_m[n] *= math.exp(lr)
        row["model_prices"] = {n: round(cum_m[n], 8) for n in names}
        row["agreement_score"] = round(
            model_output_diversity_agreement(
                list(row["model_log_returns"].values()),
                -OUTER_DAILY_ABS_CAP,
                OUTER_DAILY_ABS_CAP,
            ),
            4,
        )


def apply_cumulative_guardrail_to_path(
    path: list[dict[str, Any]],
    spot_start: float,
    vol0: float,
    training_meta: dict[str, Any] | None,
    *,
    vol_k: float = 2.0,
) -> bool:
    """
    Apply cumulative scaling to stored ensemble returns; rebuild prices.
    Returns True if guardrail fired.
    """
    if not path:
        return False
    meta = training_meta or {}
    p99_c = meta.get("historical_abs_cum_logret_14d_p99")
    rs = [float(p["predicted_log_return"]) for p in path]
    rs2, triggered = cumulative_scale_returns(
        rs,
        vol0,
        len(rs),
        historical_abs_cum14_p99=float(p99_c) if p99_c is not None else None,
    )
    if not triggered:
        return False
    rebuild_path_from_ensemble_returns(path, spot_start, rs, rs2, vol_k=vol_k)
    return True


def model_output_diversity_agreement(preds: list[float], clip_lo: float, clip_hi: float) -> float:
    """
    Agreement score that does NOT treat clip-collapsed identical outputs as high confidence.
    """
    if len(preds) < 2:
        return 0.45
    arr = np.array(preds, dtype=np.float64)
    spread = float(np.ptp(arr))
    if spread < 1e-6:
        return 0.22
    at_hi = bool(np.all(arr >= clip_hi - 1e-4))
    at_lo = bool(np.all(arr <= clip_lo + 1e-4))
    if at_hi or at_lo:
        return 0.28
    m = float(np.mean(np.abs(arr)))
    if m < 1e-10:
        return 0.25
    cv = float(np.std(arr) / m)
    return float(max(0.0, min(1.0, 1.0 - cv)))


def summarize_path_sanity(
    path: list[dict[str, Any]],
    spot_start: float,
    training_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare realized cumulative log move on the path vs training tail stats."""
    if not path or spot_start <= 0:
        return {"status": "skipped"}
    meta = training_meta or {}
    try:
        final = float(path[-1]["predicted_price"])
        cum_log = math.log(max(final, 1e-12) / float(spot_start))
    except (KeyError, TypeError, ValueError):
        return {"status": "skipped"}
    p99 = meta.get("historical_abs_cum_logret_14d_p99")
    if p99 is None or float(p99) <= 0:
        return {
            "cumulative_log_move": round(cum_log, 5),
            "status": "no_historical_baseline",
        }
    p99f = float(p99)
    ratio = abs(cum_log) / p99f
    return {
        "cumulative_log_move": round(cum_log, 5),
        "historical_abs_cum14_p99": round(p99f, 5),
        "ratio_to_historical_p99": round(ratio, 3),
        "status": "extreme_outlier" if ratio > 1.35 else "plausible",
    }