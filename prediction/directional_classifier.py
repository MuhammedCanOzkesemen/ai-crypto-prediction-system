"""
Directional classification: DOWN / NEUTRAL / UP for hybrid trading intelligence.

Labels use forward simple return over N sessions (default 5) vs a symmetric threshold
(default ±0.5%). Trained models (logistic regression + XGBoost classifier) are saved in
``directional_classifier_bundle.joblib``; legacy single ``directional_classifier.joblib`` is still supported.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# Next-N-day simple return thresholds (fraction, e.g. 0.005 = 0.5%)
DIRECTIONAL_RETURN_THRESHOLD_PCT: float = float(
    os.environ.get("DIRECTIONAL_RETURN_THRESHOLD_PCT", "0.005")
)
DIRECTIONAL_LABEL_HORIZON_DAYS: int = int(os.environ.get("DIRECTIONAL_LABEL_HORIZON_DAYS", "5"))

# Legacy: next-day log-return band for neutral (still exported for tests)
NEUTRAL_EPS_LOGRET: float = float(os.environ.get("DIRECTIONAL_NEUTRAL_EPS_LOGRET", "0.0008"))

CLASS_NAMES: tuple[str, ...] = ("down", "neutral", "up")


def build_directional_labels(y_logret: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Map next-day log returns to 0=down, 1=neutral, 2=up (legacy)."""
    e = float(NEUTRAL_EPS_LOGRET if eps is None else eps)
    y = np.asarray(y_logret, dtype=np.float64).ravel()
    return np.where(y > e, 2, np.where(y < -e, 0, 1)).astype(np.int64)


def build_directional_labels_forward_close(
    close: np.ndarray,
    horizon: int | None = None,
    threshold_pct: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Labels from **forward** N-day simple return: (close[t+N]/close[t]) - 1.

    Returns:
      y_cls: 0=down, 1=neutral, 2=up, -1=invalid (insufficient future)
      valid_mask: True where label is usable for training
    """
    h = int(DIRECTIONAL_LABEL_HORIZON_DAYS if horizon is None else horizon)
    thr = float(DIRECTIONAL_RETURN_THRESHOLD_PCT if threshold_pct is None else threshold_pct)
    c = np.asarray(close, dtype=np.float64).ravel()
    n = len(c)
    y = np.full(n, -1, dtype=np.int64)
    for i in range(max(0, n - h)):
        if c[i] <= 0 or not np.isfinite(c[i]) or not np.isfinite(c[i + h]):
            continue
        r = (c[i + h] / c[i]) - 1.0
        if r > thr:
            y[i] = 2
        elif r < -thr:
            y[i] = 0
        else:
            y[i] = 1
    valid_mask = y >= 0
    return y, valid_mask


def proba_dict_from_sklearn_row(proba_row: np.ndarray, classes: np.ndarray | list) -> dict[str, float]:
    """Map sklearn predict_proba row + classes to down/neutral/up dict."""
    labels = ("down", "neutral", "up")
    out: dict[str, float] = {"down": 0.0, "neutral": 0.0, "up": 0.0}
    for i, cls in enumerate(classes):
        idx = int(cls)
        if 0 <= idx < 3:
            out[labels[idx]] = float(proba_row[i])
    return out


def merge_directional_probabilities(p_lr: dict[str, float] | None, p_xgb: dict[str, float] | None) -> dict[str, float]:
    """Average two probability dicts; single-source fallback."""
    if p_lr and p_xgb:
        return {
            k: float((float(p_lr.get(k, 0.0)) + float(p_xgb.get(k, 0.0))) / 2.0)
            for k in ("down", "neutral", "up")
        }
    return dict(p_lr or p_xgb or {"down": 0.34, "neutral": 0.33, "up": 0.33})


def directional_confidence_from_probs(probs: dict[str, float]) -> float:
    """Max class probability in [0, 1]."""
    if not probs:
        return 0.0
    return float(max(float(probs.get("up", 0.0)), float(probs.get("down", 0.0)), float(probs.get("neutral", 0.0))))


def directional_distribution_agreement(p1: dict[str, float] | None, p2: dict[str, float] | None) -> float:
    """
    Agreement between two categorical distributions in [0, 1].
    1 = identical; 0 = maximally different on the 3-simplex.
    """
    if not p1 or not p2:
        return 1.0
    keys = ("up", "down", "neutral")
    d = sum(abs(float(p1.get(k, 0.0)) - float(p2.get(k, 0.0))) for k in keys)
    return float(max(0.0, min(1.0, 1.0 - 0.5 * d)))


def regression_agreement_with_diagnostics(
    lr_matrix: list[list[float]] | None,
    model_names: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Robust regression agreement from per-step model log returns.

    Unlike a plain CV over all model outputs, this treats ``2 models clustered + 1 outlier``
    as better than ``3 models split in sign`` and emits diagnostics describing the diverging
    model so downstream trade logic can stay strict without being blindly pessimistic.
    """
    if not lr_matrix:
        return 0.0, {
            "score": 0.0,
            "same_sign_fraction": 0.0,
            "sign_consensus_mean": 0.0,
            "magnitude_consensus_mean": 0.0,
            "diverging_model": None,
            "diverging_model_score": 0.0,
            "per_model_divergence": {},
            "per_model_sign_mismatch_fraction": {},
            "per_model_mean_abs_dev_ratio": {},
        }

    first_n = len(lr_matrix[0]) if lr_matrix and lr_matrix[0] else 0
    names = list(model_names or [])
    if len(names) != first_n:
        names = [f"model_{i + 1}" for i in range(first_n)]

    step_scores: list[float] = []
    sign_scores: list[float] = []
    mag_scores: list[float] = []
    same_sign_steps = 0
    dev_sums = np.zeros(first_n, dtype=np.float64)
    sign_mismatch = np.zeros(first_n, dtype=np.float64)
    outlier_hits = np.zeros(first_n, dtype=np.float64)
    used_steps = 0

    for row in lr_matrix:
        arr = np.asarray(row, dtype=np.float64)
        if len(arr) != first_n or len(arr) < 2 or not np.all(np.isfinite(arr)):
            continue
        used_steps += 1
        center = float(np.median(arr))
        scale_ref = max(abs(center), float(np.mean(np.abs(arr))), 0.0025)
        dev_ratio = np.abs(arr - center) / scale_ref
        mean_dev = float(np.mean(dev_ratio))
        mag_consensus = float(1.0 / (1.0 + mean_dev / 0.85))

        pos = float(np.mean(arr > 1e-6))
        neg = float(np.mean(arr < -1e-6))
        flat = max(0.0, 1.0 - pos - neg)
        sign_consensus = float(max(pos, neg, 0.80 * flat))
        mixed_sign = pos > 0.0 and neg > 0.0
        if not mixed_sign:
            same_sign_steps += 1

        step_score = 0.45 * sign_consensus + 0.55 * mag_consensus
        if mixed_sign:
            step_score *= 0.74
        if float(np.max(dev_ratio)) >= 1.35 and float(np.mean(dev_ratio <= 0.85)) >= 0.34:
            # Single-model divergence should hurt less than broad disagreement.
            step_score = min(1.0, step_score * 1.06)
        step_scores.append(float(max(0.0, min(1.0, step_score))))
        sign_scores.append(sign_consensus)
        mag_scores.append(mag_consensus)

        majority_sign = 0
        if pos > neg and pos >= 0.5:
            majority_sign = 1
        elif neg > pos and neg >= 0.5:
            majority_sign = -1
        step_max = float(np.max(dev_ratio))
        for i, v in enumerate(arr):
            dev_sums[i] += float(dev_ratio[i])
            if majority_sign != 0:
                s = 1 if v > 1e-6 else -1 if v < -1e-6 else 0
                if s != 0 and s != majority_sign:
                    sign_mismatch[i] += 1.0
            if float(dev_ratio[i]) >= max(1.15, 0.88 * step_max):
                outlier_hits[i] += 1.0

    if used_steps == 0:
        return 0.0, {
            "score": 0.0,
            "same_sign_fraction": 0.0,
            "sign_consensus_mean": 0.0,
            "magnitude_consensus_mean": 0.0,
            "diverging_model": None,
            "diverging_model_score": 0.0,
            "per_model_divergence": {},
            "per_model_sign_mismatch_fraction": {},
            "per_model_mean_abs_dev_ratio": {},
        }

    avg_dev = dev_sums / float(used_steps)
    mismatch_frac = sign_mismatch / float(used_steps)
    outlier_frac = outlier_hits / float(used_steps)
    per_model_divergence: dict[str, float] = {}
    per_model_mean_abs_dev_ratio: dict[str, float] = {}
    per_model_sign_mismatch_fraction: dict[str, float] = {}
    for i, name in enumerate(names):
        dev_score = min(1.0, float(avg_dev[i]) / 1.25)
        score = 0.62 * dev_score + 0.24 * float(mismatch_frac[i]) + 0.14 * float(outlier_frac[i])
        per_model_divergence[name] = round(float(max(0.0, min(1.0, score))), 4)
        per_model_mean_abs_dev_ratio[name] = round(float(avg_dev[i]), 4)
        per_model_sign_mismatch_fraction[name] = round(float(mismatch_frac[i]), 4)

    diverging_model = max(per_model_divergence, key=per_model_divergence.get) if per_model_divergence else None
    diverging_score = float(per_model_divergence.get(diverging_model, 0.0)) if diverging_model else 0.0
    if diverging_score < 0.26:
        diverging_model = None
        diverging_score = 0.0

    score = float(np.mean(step_scores))
    return float(round(max(0.0, min(1.0, score)), 4)), {
        "score": round(float(score), 4),
        "same_sign_fraction": round(float(same_sign_steps / float(used_steps)), 4),
        "sign_consensus_mean": round(float(np.mean(sign_scores)), 4),
        "magnitude_consensus_mean": round(float(np.mean(mag_scores)), 4),
        "diverging_model": diverging_model,
        "diverging_model_score": round(diverging_score, 4),
        "per_model_divergence": per_model_divergence,
        "per_model_sign_mismatch_fraction": per_model_sign_mismatch_fraction,
        "per_model_mean_abs_dev_ratio": per_model_mean_abs_dev_ratio,
    }


def agreement_trend_summary(agreements: list[float] | None) -> dict[str, Any]:
    """Agreement quality across the forecast horizon: stable/improving/collapsing."""
    if not agreements:
        return {"score": 0.5, "label": "mixed", "slope": 0.0, "stability": 0.5}
    arr = np.asarray([float(x) for x in agreements], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"score": 0.5, "label": "mixed", "slope": 0.0, "stability": 0.5}
    if len(arr) == 1:
        v = float(max(0.0, min(1.0, arr[0])))
        return {"score": round(v, 4), "label": "stable", "slope": 0.0, "stability": round(v, 4)}
    slope = float(arr[-1] - arr[0])
    std = float(np.std(arr))
    mean_level = float(np.mean(arr))
    stability = float(max(0.0, min(1.0, 1.0 - min(1.0, std / 0.18))))
    slope_n = float(max(-1.0, min(1.0, slope / 0.18)))
    trend_signal = 0.5 + 0.5 * slope_n
    score = float(max(0.0, min(1.0, 0.5 * stability + 0.3 * trend_signal + 0.2 * mean_level)))
    if slope <= -0.10 or (arr[-1] < mean_level - 0.08 and std > 0.06):
        label = "collapsing"
    elif slope >= 0.08:
        label = "improving"
    elif stability >= 0.72:
        label = "stable"
    else:
        label = "mixed"
    return {
        "score": round(score, 4),
        "label": label,
        "slope": round(slope, 4),
        "stability": round(stability, 4),
    }


def combined_agreement_score(regression_agreement: float, directional_model_agreement: float) -> float:
    """Blend robust path agreement with directional-model agreement."""
    ra = float(max(0.0, min(1.0, regression_agreement)))
    da = float(max(0.0, min(1.0, directional_model_agreement)))
    out = 0.67 * ra + 0.33 * da
    if ra >= 0.60 and da >= 0.60:
        out = min(1.0, out + 0.05 * min(ra, da))
    elif ra < 0.35 and da < 0.55:
        out *= 0.92
    return float(max(0.0, min(1.0, out)))


def compute_hybrid_primary_confidence(
    directional_confidence: float,
    combined_agreement_score_val: float,
    signal_strength_score: float,
) -> float:
    """Primary confidence: directional + combined agreement + signal (no legacy financial stack)."""
    dc = float(max(0.0, min(1.0, directional_confidence)))
    ca = float(max(0.0, min(1.0, combined_agreement_score_val)))
    ss = float(max(0.0, min(1.0, signal_strength_score)))
    return float(np.clip(0.4 * dc + 0.3 * ca + 0.3 * ss, 0.0, 1.0))


BUNDLE_VERSION = 2
