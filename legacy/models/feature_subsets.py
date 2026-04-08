"""
Per-model feature views for ensemble diversity (same scaler, different column subsets).

- random_forest: volatility + momentum heavy (no interaction-only columns).
- xgboost: interactions + trend-strength structure (+ returns/ROC for direction).
- lightgbm: full feature set.

Indices match ``get_feature_columns(include_targets=False)`` order at train/infer time.
"""

from __future__ import annotations

from typing import Any


def _min_columns(n_feat: int) -> int:
    return max(32, min(n_feat, int(n_feat * 0.58)))


def _top_up(picked: list[str], feat_order: list[str], *, min_n: int) -> None:
    have = set(picked)
    for c in feat_order:
        if len(picked) >= min_n:
            break
        if c not in have:
            picked.append(c)
            have.add(c)


def _is_vol_or_momentum_column(c: str) -> bool:
    """RandomForest: vol, bands, returns, MACD/RSI/ROC, lags, breakouts, range."""
    if c.startswith("interact_"):
        return False
    needles = (
        "atr_14",
        "rolling_volatility_14",
        "ewma_vol",
        "rolling_std_logret",
        "vol_regime_encoded",
        "vol_expansion",
        "compression_expansion",
        "bb_",
        "macd_",
        "rsi_14",
        "roc_",
        "daily_return",
        "return_",
        "log_return_1d",
        "log_return_accel",
        "momentum_logret",
        "momentum_change_rate",
        "ema_slope_accel",
        "volume_change",
        "breakout_",
        "dist_to_roll",
        "range_position",
        "log_close",
        "trend_consistency",
        "twitter_",
        "sentiment_x_",
    )
    return any(x in c for x in needles)


def _is_interaction_or_trend_column(c: str) -> bool:
    """XGBoost: interactions, ADX/DI, EMA/SMA structure, trend consistency, slope accel."""
    if c.startswith("interact_") or c.startswith("sentiment_x_") or c.startswith("twitter_"):
        return True
    exact = {
        "trend_strength_adx",
        "ema_cross_norm",
        "trend_consistency_score",
        "ema_slope_accel",
        "momentum_change_rate",
        "range_position_14d_window",
        "compression_expansion_score",
    }
    if c in exact:
        return True
    if c.startswith("adx_") or c.startswith("plus_di_") or c.startswith("minus_di_"):
        return True
    if c in ("ema_20", "ema_50", "sma_20", "sma_50"):
        return True
    # Directional context without full vol stack
    if c.startswith("return_") or c.startswith("roc_") or c in ("daily_return", "momentum_logret_7d", "momentum_logret_14d"):
        return True
    return False


def feature_columns_for_model(model_name: str, feat_order: list[str]) -> list[str]:
    key = model_name.lower().strip()
    if key == "lightgbm":
        return list(feat_order)
    if key == "random_forest":
        picked = [c for c in feat_order if _is_vol_or_momentum_column(c)]
        _top_up(picked, feat_order, min_n=_min_columns(len(feat_order)))
        return picked
    if key == "xgboost":
        picked = [c for c in feat_order if _is_interaction_or_trend_column(c)]
        _top_up(picked, feat_order, min_n=_min_columns(len(feat_order)))
        return picked
    return list(feat_order)


def per_model_feature_lists(model_names: list[str], feat_order: list[str]) -> dict[str, list[str]]:
    return {name: feature_columns_for_model(name, feat_order) for name in model_names}


def indices_dict_for_metadata(per_model_cols: dict[str, list[str]], feat_order: list[str]) -> dict[str, list[int]]:
    name_to_i = {c: j for j, c in enumerate(feat_order)}
    out: dict[str, list[int]] = {}
    for name, cols in per_model_cols.items():
        out[name] = [name_to_i[c] for c in cols if c in name_to_i]
    return out


def load_indices_from_metadata(meta: dict[str, Any] | None, model_name: str, n_full: int) -> list[int] | None:
    if not meta:
        return None
    raw = meta.get("per_model_feature_indices")
    if not isinstance(raw, dict):
        return None
    idx = raw.get(model_name)
    if not isinstance(idx, list) or not idx:
        return None
    out = [int(i) for i in idx if 0 <= int(i) < n_full]
    return out if out else None
