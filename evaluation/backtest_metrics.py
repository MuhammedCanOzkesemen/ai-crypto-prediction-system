"""Portfolio metrics for trade backtests (no fabricated values; null when undefined)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(np.mean(xs))


def _safe_median(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(np.median(xs))


def compute_trade_statistics(
    trade_returns: list[float],
    equity_daily: np.ndarray | None = None,
    daily_returns: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Aggregated strategy metrics.

    ``trade_returns``: per-trade simple returns (fractional, e.g. 0.01 = +1%).
    ``equity_daily``: optional daily equity levels (for drawdown / Sharpe on days).
    ``daily_returns``: optional aligned daily simple returns; if None, derived from equity_daily.
    """
    n = len(trade_returns)
    out: dict[str, Any] = {
        "total_trades": n,
        "win_rate": None,
        "avg_return_per_trade": None,
        "median_return_per_trade": None,
        "total_return": None,
        "compounded_return": None,
        "max_drawdown": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "profit_factor": None,
        "expectancy": None,
        "average_holding_days": None,
        "exposure_ratio": None,
        "best_trade": None,
        "worst_trade": None,
        "metrics_notes": [],
    }

    if n == 0:
        out["metrics_notes"].append("No completed trades; trade-based metrics are null.")
        if equity_daily is not None and len(equity_daily) > 1:
            er = np.diff(equity_daily) / np.clip(equity_daily[:-1], 1e-12, None)
            out["sharpe_ratio"], out["sortino_ratio"], out["max_drawdown"] = _equity_curve_metrics(
                equity_daily, er
            )
        return out

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r < 0]
    out["win_rate"] = float(len(wins) / n)
    out["avg_return_per_trade"] = _safe_mean(trade_returns)
    out["median_return_per_trade"] = _safe_median(trade_returns)
    out["total_return"] = float(np.sum(trade_returns))
    prod = float(np.prod([1.0 + r for r in trade_returns]))
    out["compounded_return"] = prod - 1.0
    out["expectancy"] = float(out["avg_return_per_trade"]) if out["avg_return_per_trade"] is not None else None
    out["best_trade"] = float(max(trade_returns))
    out["worst_trade"] = float(min(trade_returns))

    gp = float(sum(wins)) if wins else 0.0
    gl = float(abs(sum(losses))) if losses else 0.0
    if gl < 1e-15:
        out["profit_factor"] = None
        out["metrics_notes"].append("Profit factor undefined (no losing trades).")
    else:
        out["profit_factor"] = gp / gl

    if equity_daily is not None and len(equity_daily) > 1:
        er = daily_returns
        if er is None:
            er = np.diff(equity_daily) / np.clip(equity_daily[:-1], 1e-12, None)
        sh, so, dd = _equity_curve_metrics(equity_daily, er)
        out["sharpe_ratio"] = sh
        out["sortino_ratio"] = so
        out["max_drawdown"] = dd
    else:
        out["metrics_notes"].append("No daily equity series; Sharpe/Sortino/drawdown from equity are null.")

    if n < 2:
        out["metrics_notes"].append("Sharpe/Sortino on trade returns require daily equity curve; with <2 trades use caution.")

    return out


def _equity_curve_metrics(equity: np.ndarray, daily_r: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if len(daily_r) < 2:
        return None, None, None
    mu = float(np.mean(daily_r))
    sd = float(np.std(daily_r, ddof=1))
    sharpe = None
    if sd > 1e-12:
        sharpe = float((mu / sd) * math.sqrt(252))

    neg = daily_r[daily_r < 0]
    if len(neg) < 2:
        sortino = None
    else:
        ddn = float(np.std(neg, ddof=1))
        sortino = float((mu / ddn) * math.sqrt(252)) if ddn > 1e-12 else None

    peak = np.maximum.accumulate(equity)
    dd = (equity / np.clip(peak, 1e-12, None)) - 1.0
    max_dd = float(np.min(dd)) if len(dd) else None
    return sharpe, sortino, max_dd


def regime_performance_summary(
    trades_by_regime: dict[str, list[float]],
) -> dict[str, dict[str, Any]]:
    """For each regime label: n_trades, win_rate, avg_return, total_return."""
    out: dict[str, dict[str, Any]] = {}
    for reg, rets in trades_by_regime.items():
        n = len(rets)
        if n == 0:
            out[reg] = {
                "trades": 0,
                "win_rate": None,
                "avg_return": None,
                "total_return": None,
            }
            continue
        wins = sum(1 for r in rets if r > 0)
        out[reg] = {
            "trades": n,
            "win_rate": float(wins / n),
            "avg_return": float(np.mean(rets)),
            "total_return": float(np.sum(rets)),
        }
    return out


def pick_best_regime(regime_summary: dict[str, dict[str, Any]]) -> str | None:
    best = None
    best_tot = None
    for reg, d in regime_summary.items():
        t = d.get("total_return")
        if t is None:
            continue
        if best_tot is None or t > best_tot:
            best_tot = t
            best = reg
    return best
