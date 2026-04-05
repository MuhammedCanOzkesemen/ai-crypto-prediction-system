"""
Walk-forward trade backtest: chronological simulation, no future leakage in features.

Decision at bar *t* uses only ``ohlc.iloc[: t + 1]``. Entry is simulated at the open of bar *t+1*.
Trade exits use realized future OHLC only *after* entry (standard realized-path simulation).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
from unittest.mock import patch

from evaluation.backtest_metrics import (
    compute_trade_statistics,
    pick_best_regime,
    regime_performance_summary,
)
from evaluation.trade_execution import simulate_long_trade_daily_ohlc
from prediction.decision_layer import clear_signal_history
from prediction.predictor import Predictor
from utils.config import settings
from utils.logging_setup import get_logger

logger = get_logger(__name__)

REQ = ["date", "open", "high", "low", "close", "volume"]
MIN_HISTORY = 80
ACTIONABLE_DECISIONS = ("STRONG_BUY", "WEAK_BUY", "PROBING_BUY")


def backtests_dir() -> Path:
    p = settings.data.artifact_dir / "backtests"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slug(coin: str) -> str:
    return coin.replace(" ", "_")


def load_ohlcv_frame(coin: str, features_dir: Path | None = None) -> pd.DataFrame | None:
    """Load feature parquet OHLCV columns (same source as training)."""
    fd = Path(features_dir) if features_dir else settings.training.features_dir
    path = fd / f"{_slug(coin)}_features.parquet"
    if not path.exists():
        logger.warning("Missing features parquet: %s", path)
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.exception("Failed to read %s: %s", path, e)
        return None
    if not all(c in df.columns for c in REQ):
        return None
    out = df[REQ].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def load_feature_parquet_frame(coin: str, features_dir: Path | None = None) -> pd.DataFrame | None:
    """Load the raw feature parquet so live-style inference can be replayed on a truncated slice."""
    fd = Path(features_dir) if features_dir else settings.training.features_dir
    path = fd / f"{_slug(coin)}_features.parquet"
    if not path.exists():
        logger.warning("Missing features parquet: %s", path)
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logger.exception("Failed to read %s: %s", path, e)
        return None
    if "date" not in df.columns:
        logger.warning("Feature parquet missing date column: %s", path)
        return None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def _bucket_rejection(dec_reasons: list[str], trade_reasons: list[str]) -> str | None:
    blob = " ".join(dec_reasons + trade_reasons).lower()
    if "cooldown" in blob:
        return "cooldown"
    if "sanity" in blob:
        return "feature_sanity"
    if "shock" in blob:
        return "volatility_shock"
    if "confidence" in blob and "below" in blob:
        return "low_confidence"
    if "agreement" in blob:
        return "low_agreement"
    if "edge" in blob:
        return "low_edge"
    if "regime" in blob and "volatile" in blob:
        return "regime_filter"
    if "trend" in blob or "neutral" in blob or "aligned" in blob:
        return "trend_filter"
    if "sentiment" in blob or "conflict" in blob:
        return "sentiment_conflict"
    return "other"


def _safe_avg(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(np.mean(xs))


def _extract_decision_snapshot(raw: dict[str, Any] | None) -> dict[str, Any]:
    diag = (raw or {}).get("_forecast_diagnostics") or {}
    return {
        "confidence_score": float(diag.get("confidence_score", 0.0) or 0.0),
        "directional_confidence": float(diag.get("directional_confidence", 0.0) or 0.0),
        "combined_agreement_score": float(diag.get("combined_agreement_score", 0.0) or 0.0),
        "trade_decision": str(diag.get("trade_decision", "NO_TRADE")),
        "decision_blockers": dict(diag.get("decision_blockers") or {}),
        "trade_reasons": list(diag.get("trade_reasons") or []),
        "decision_summary": dict(diag.get("decision_summary") or {}),
    }


def _prime_decision_state(
    predictor: Predictor,
    coin: str,
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    *,
    include_twitter: bool,
    include_regime_filter: bool,
    decision_mode: str,
    include_cooldown: bool,
) -> bool:
    """
    Replay prior dates so decision history and cooldown state match the walk-forward backtest.
    Returns whether cooldown should be active on ``target_date``.
    """
    clear_signal_history(coin)
    cd_days = int(os.environ.get("TRADE_COOLDOWN_DAYS", "3"))
    cooldown_until: date | None = None

    for idx in range(len(df)):
        ts = pd.Timestamp(df.iloc[idx]["date"]).normalize()
        if ts >= target_date:
            break
        if idx < MIN_HISTORY - 1:
            continue
        d_only = ts.date()
        cooldown_active_now = bool(include_cooldown and cooldown_until is not None and d_only <= cooldown_until)
        hist = df.iloc[: idx + 1].copy()
        raw = predictor.predict_for_backtest(
            coin,
            hist,
            include_twitter=include_twitter,
            include_regime_filter=include_regime_filter,
            cooldown_active_override=cooldown_active_now,
            register_cooldown=False,
            decision_mode=decision_mode,
        )
        snapshot = _extract_decision_snapshot(raw)
        if include_cooldown and snapshot["trade_decision"] in ACTIONABLE_DECISIONS:
            cooldown_until = d_only + timedelta(days=cd_days)

    return bool(include_cooldown and cooldown_until is not None and target_date.date() <= cooldown_until)


def _execution_profile_for_decision(decision: str) -> tuple[float, float, int]:
    if decision == "STRONG_BUY":
        return 0.02, 0.03, 3
    # PROBING_BUY is a lighter conviction tier, so reuse the shortest existing weak profile.
    return 0.015, 0.02, 1


def compare_live_and_backtest_prediction(
    coin: str,
    as_of_date: str,
    *,
    decision_mode: str = "trade_decision",
    include_twitter: bool = True,
    include_regime_filter: bool = True,
    include_cooldown: bool = True,
    features_dir: Path | None = None,
    models_dir: Path | None = None,
    evaluation_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Run the live entrypoint and backtest entrypoint against the same truncated history and compare
    the core decision diagnostics. Mismatches are logged with detailed context.
    """
    features_dir = Path(features_dir) if features_dir else settings.training.features_dir
    target_date = pd.Timestamp(as_of_date).normalize()
    full_feature_df = load_feature_parquet_frame(coin, features_dir)
    if full_feature_df is None or full_feature_df.empty:
        return {
            "coin": coin,
            "as_of_date": target_date.strftime("%Y-%m-%d"),
            "error": "missing_feature_parquet",
        }

    date_series = pd.to_datetime(full_feature_df["date"]).dt.normalize()
    truncated_feature_df = full_feature_df.loc[date_series <= target_date].copy()
    if truncated_feature_df.empty:
        return {
            "coin": coin,
            "as_of_date": target_date.strftime("%Y-%m-%d"),
            "error": "no_rows_before_date",
        }
    if not all(c in truncated_feature_df.columns for c in REQ):
        return {
            "coin": coin,
            "as_of_date": target_date.strftime("%Y-%m-%d"),
            "error": "missing_ohlcv_columns",
        }

    hist = truncated_feature_df[REQ].copy().reset_index(drop=True)
    if len(hist) < MIN_HISTORY:
        return {
            "coin": coin,
            "as_of_date": target_date.strftime("%Y-%m-%d"),
            "error": "insufficient_history",
            "rows": len(hist),
        }

    predictor = Predictor(models_dir=models_dir, evaluation_dir=evaluation_dir)
    if not predictor.load_models(coin):
        return {
            "coin": coin,
            "as_of_date": target_date.strftime("%Y-%m-%d"),
            "error": "no_models",
        }

    cooldown_active_now = _prime_decision_state(
        predictor,
        coin,
        hist,
        target_date,
        include_twitter=include_twitter,
        include_regime_filter=include_regime_filter,
        decision_mode=decision_mode,
        include_cooldown=include_cooldown,
    )

    clear_signal_history(coin)
    _prime_decision_state(
        predictor,
        coin,
        hist,
        target_date,
        include_twitter=include_twitter,
        include_regime_filter=include_regime_filter,
        decision_mode=decision_mode,
        include_cooldown=include_cooldown,
    )
    with TemporaryDirectory() as tmpdir_s:
        tmpdir = Path(tmpdir_s)
        temp_path = tmpdir / f"{_slug(coin)}_features.parquet"
        truncated_feature_df.to_parquet(temp_path, index=False)
        with patch("prediction.predictor.cooldown_active", return_value=(cooldown_active_now, None)):
            with patch("prediction.predictor.register_actionable_signal", return_value=None):
                live_raw = predictor.predict_from_latest_features(coin, features_dir=tmpdir)
    live_sig = _extract_decision_snapshot(live_raw)

    clear_signal_history(coin)
    _prime_decision_state(
        predictor,
        coin,
        hist,
        target_date,
        include_twitter=include_twitter,
        include_regime_filter=include_regime_filter,
        decision_mode=decision_mode,
        include_cooldown=include_cooldown,
    )
    backtest_raw = predictor.predict_for_backtest(
        coin,
        hist,
        include_twitter=include_twitter,
        include_regime_filter=include_regime_filter,
        cooldown_active_override=cooldown_active_now,
        register_cooldown=False,
        decision_mode=decision_mode,
    )
    backtest_sig = _extract_decision_snapshot(backtest_raw)

    keys = ("confidence_score", "directional_confidence", "combined_agreement_score", "trade_decision")
    mismatches: dict[str, Any] = {}
    for key in keys:
        lv = live_sig.get(key)
        bv = backtest_sig.get(key)
        if isinstance(lv, float) or isinstance(bv, float):
            if abs(float(lv) - float(bv)) > 1e-9:
                mismatches[key] = {"live": lv, "backtest": bv}
        elif lv != bv:
            mismatches[key] = {"live": lv, "backtest": bv}

    if mismatches:
        logger.warning(
            "Live/backtest mismatch for %s on %s: %s | live=%s | backtest=%s",
            coin,
            target_date.strftime("%Y-%m-%d"),
            mismatches,
            live_sig,
            backtest_sig,
        )
    else:
        logger.info("Live/backtest parity confirmed for %s on %s", coin, target_date.strftime("%Y-%m-%d"))

    return {
        "coin": coin,
        "as_of_date": target_date.strftime("%Y-%m-%d"),
        "cooldown_active": cooldown_active_now,
        "matches": not mismatches,
        "mismatches": mismatches,
        "live": live_sig,
        "backtest": backtest_sig,
    }


def run_trade_backtest(
    coin: str,
    start_date: str,
    end_date: str,
    *,
    decision_mode: str = "trade_decision",
    include_twitter: bool = True,
    include_regime_filter: bool = True,
    include_cooldown: bool = True,
    features_dir: Path | None = None,
    models_dir: Path | None = None,
    evaluation_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Walk-forward simulation over [start_date, end_date] inclusive (calendar filter on bar dates).

    See module docstring for leakage policy. When ``include_cooldown`` is True, cooldown is
    simulated from trade dates (disk JSON is not read for gating).
    """
    features_dir = Path(features_dir) if features_dir else settings.training.features_dir
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    df = load_ohlcv_frame(coin, features_dir)
    if df is None or len(df) < MIN_HISTORY:
        return {
            "coin": coin,
            "error": "insufficient_data",
            "message": "Need features parquet with OHLCV and at least 80 rows.",
        }

    predictor = Predictor(models_dir=models_dir, evaluation_dir=evaluation_dir)
    if not predictor.load_models(coin):
        return {"coin": coin, "error": "no_models", "message": f"No trained models for {coin}."}

    clear_signal_history(coin)
    cd_days = int(os.environ.get("TRADE_COOLDOWN_DAYS", "3"))
    cooldown_until: date | None = None

    cash_equity = 1.0
    pos: dict[str, Any] | None = None

    trade_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    health: dict[str, Any] = {
        "days_evaluated": 0,
        "no_trade_days": 0,
        "strong_buy_signals": 0,
        "weak_buy_signals": 0,
        "probing_buy_signals": 0,
        "actionable_signal_days": 0,
        "blocked_cooldown_days": 0,
        "rejection_buckets": defaultdict(int),
        "signal_dates": [],
    }

    regime_returns: dict[str, list[float]] = defaultdict(list)
    strong_count = weak_count = probing_count = 0
    last_signal_idx: int | None = None
    signal_gaps: list[int] = []

    peak_equity = 1.0
    in_pos_days = 0

    for idx in range(len(df)):
        ts = pd.Timestamp(df.iloc[idx]["date"]).normalize()
        if ts < start or ts > end:
            continue
        if idx < MIN_HISTORY - 1:
            continue

        health["days_evaluated"] += 1
        close_px = float(df.iloc[idx]["close"])
        hist = df.iloc[: idx + 1].copy()
        d_only = pd.Timestamp(hist["date"].iloc[-1]).date()

        cd_active = False
        if include_cooldown and cooldown_until is not None and d_only <= cooldown_until:
            cd_active = True
        cooldown_override = cd_active if include_cooldown else False

        raw = predictor.predict_for_backtest(
            coin,
            hist,
            include_twitter=include_twitter,
            include_regime_filter=include_regime_filter,
            cooldown_active_override=cooldown_override,
            register_cooldown=False,
            decision_mode=decision_mode,
        )

        td = "NO_TRADE"
        diag: dict[str, Any] = {}
        if raw is not None:
            diag = raw.get("_forecast_diagnostics") or {}
            td = str(diag.get("trade_decision", "NO_TRADE"))

        if bool(diag.get("signal_cooldown_active")):
            health["blocked_cooldown_days"] += 1

        if td == "STRONG_BUY":
            health["strong_buy_signals"] += 1
        elif td == "WEAK_BUY":
            health["weak_buy_signals"] += 1
        elif td == "PROBING_BUY":
            health["probing_buy_signals"] += 1

        if td in ACTIONABLE_DECISIONS:
            health["actionable_signal_days"] += 1
            health["signal_dates"].append(ts.isoformat())
            if last_signal_idx is not None:
                signal_gaps.append(idx - last_signal_idx)
            last_signal_idx = idx

        if td == "NO_TRADE":
            health["no_trade_days"] += 1
            dr = list(diag.get("decision_rejection_reasons") or [])
            tr = list(diag.get("trade_reasons") or [])
            b = _bucket_rejection(dr, tr)
            if b:
                health["rejection_buckets"][b] += 1

        was_in_pos = pos is not None
        if pos is not None:
            in_pos_days += 1
            if idx == int(pos["exit_idx"]):
                cash_equity = float(pos["shares"]) * float(pos["exit_price"])
                eq = cash_equity
                pos = None
            else:
                eq = float(pos["shares"]) * close_px
        else:
            eq = cash_equity

        peak_equity = max(peak_equity, eq)
        dd = (eq / peak_equity) - 1.0 if peak_equity > 0 else 0.0

        in_flag = 1 if was_in_pos else 0
        equity_rows.append({
            "date": ts.isoformat(),
            "equity": round(eq, 8),
            "drawdown": round(dd, 8),
            "in_position": in_flag,
            "active_decision": td,
        })

        can_open = pos is None
        if can_open and raw is not None and td in ACTIONABLE_DECISIONS:
            entry_i = idx + 1
            if entry_i < len(df):
                sl_pct, tp_pct, max_bars = _execution_profile_for_decision(td)
                sim = simulate_long_trade_daily_ohlc(
                    df, entry_i, stop_loss_pct=sl_pct, take_profit_pct=tp_pct, max_holding_bars=max_bars
                )
                if (
                    sim.get("exit_reason") != "INVALID"
                    and sim.get("return_pct") is not None
                    and sim.get("entry_price") is not None
                    and sim.get("exit_price") is not None
                ):
                    entry_px = float(sim["entry_price"])
                    exit_px = float(sim["exit_price"])
                    ret = float(sim["return_pct"])
                    shares = cash_equity / entry_px
                    if td == "STRONG_BUY":
                        strong_count += 1
                    elif td == "WEAK_BUY":
                        weak_count += 1
                    else:
                        probing_count += 1
                    reg = str(diag.get("market_regime", "RANGING"))
                    regime_returns[reg].append(ret)
                    exit_idx = int(sim["exit_row_index"])
                    trade_rows.append({
                        "decision_date": _iso_date(hist["date"].iloc[-1]),
                        "entry_date": _iso_date(df.iloc[entry_i]["date"]),
                        "exit_date": _iso_date(df.iloc[exit_idx]["date"]),
                        "coin": coin,
                        "decision": td,
                        "market_regime": reg,
                        "confidence_score": diag.get("confidence_score"),
                        "edge_score": diag.get("edge_score"),
                        "agreement": raw.get("_mean_path_agreement"),
                        "signal_strength_score": diag.get("signal_strength_score"),
                        "consensus_score": diag.get("consensus_score"),
                        "stability_score": diag.get("stability_score"),
                        "trend_confirmation_score": diag.get("trend_confirmation_score"),
                        "twitter_sentiment_used": bool(diag.get("twitter_sentiment_used", False)),
                        "sentiment_alignment": diag.get("sentiment_alignment"),
                        "entry_price": entry_px,
                        "exit_price": exit_px,
                        "return_pct": ret,
                        "holding_days": sim.get("holding_days"),
                        "exit_reason": sim.get("exit_reason"),
                        "trade_reasons": json.dumps(diag.get("trade_reasons") or []),
                    })
                    pos = {
                        "shares": shares,
                        "entry_idx": entry_i,
                        "exit_idx": exit_idx,
                        "exit_price": exit_px,
                    }
                    if include_cooldown:
                        cooldown_until = d_only + timedelta(days=cd_days)

    de = max(1, health["days_evaluated"])
    rej = dict(health["rejection_buckets"])
    health_summary = {
        "signal_frequency": health["actionable_signal_days"] / de,
        "strong_signal_frequency": health["strong_buy_signals"] / de,
        "weak_signal_frequency": health["weak_buy_signals"] / de,
        "probing_signal_frequency": health["probing_buy_signals"] / de,
        "average_time_between_signals": float(np.mean(signal_gaps)) if signal_gaps else None,
        "percentage_days_blocked_by_cooldown": health["blocked_cooldown_days"] / de,
        "percentage_no_trade": health["no_trade_days"] / de,
        "rejection_percentages": {k: v / de for k, v in rej.items()},
        "rejection_counts": rej,
    }

    trade_returns = [float(r["return_pct"]) for r in trade_rows if r.get("return_pct") is not None]
    eq_arr = np.array([r["equity"] for r in equity_rows], dtype=float)
    er = np.diff(eq_arr) / np.clip(eq_arr[:-1], 1e-12, None) if len(eq_arr) > 1 else None

    stats = compute_trade_statistics(trade_returns, equity_daily=eq_arr if len(eq_arr) else None, daily_returns=er)
    stats["strong_buy_trades"] = strong_count
    stats["weak_buy_trades"] = weak_count
    stats["probing_buy_trades"] = probing_count
    stats["no_trade_days"] = health["no_trade_days"]
    stats["average_holding_days"] = _safe_avg([float(r["holding_days"]) for r in trade_rows if r.get("holding_days") is not None])
    stats["exposure_ratio"] = float(in_pos_days / de) if de else 0.0

    regime_breakdown = regime_performance_summary(dict(regime_returns))
    stats["regime_breakdown"] = regime_breakdown
    stats["best_regime"] = pick_best_regime(regime_breakdown)

    out_dir = backtests_dir()
    slug = _slug(coin)
    trades_path = out_dir / f"{slug}_trades.csv"
    equity_path = out_dir / f"{slug}_equity_curve.csv"
    summary_path = out_dir / f"{slug}_summary.json"

    cols = [
        "decision_date", "entry_date", "exit_date", "coin", "decision", "market_regime",
        "confidence_score", "edge_score", "agreement", "signal_strength_score",
        "consensus_score", "stability_score", "trend_confirmation_score",
        "twitter_sentiment_used", "sentiment_alignment",
        "entry_price", "exit_price", "return_pct", "holding_days", "exit_reason", "trade_reasons",
    ]
    if trade_rows:
        pd.DataFrame(trade_rows).to_csv(trades_path, index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(trades_path, index=False)

    pd.DataFrame(equity_rows).to_csv(equity_path, index=False)

    summary_payload = {
        "coin": coin,
        "start_date": start_date,
        "end_date": end_date,
        "decision_mode": decision_mode,
        "include_twitter": include_twitter,
        "include_regime_filter": include_regime_filter,
        "include_cooldown": include_cooldown,
        "artifacts": {
            "trades_csv": str(trades_path.resolve()),
            "equity_curve_csv": str(equity_path.resolve()),
            "summary_json": str(summary_path.resolve()),
        },
        "metrics": stats,
        "strategy_health": health_summary,
        "execution_assumptions": {
            "bar_frequency": "daily",
            "entry": "next_bar_open_after_decision",
            "stop_take_profit": "intraday_high_low; if both hit same bar assume_stop_first",
            "strong_buy": {"stop_loss_pct": 0.02, "take_profit_pct": 0.03, "max_holding_sessions": 3},
            "weak_buy": {"stop_loss_pct": 0.015, "take_profit_pct": 0.02, "max_holding_sessions": 1},
            "probing_buy": {"stop_loss_pct": 0.015, "take_profit_pct": 0.02, "max_holding_sessions": 1},
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, default=str)

    return summary_payload


def _iso_date(v: Any) -> str:
    ts = pd.Timestamp(v)
    return ts.isoformat()
