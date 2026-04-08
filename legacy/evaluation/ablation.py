"""
Controlled ablation runs: same date range and coin, different pipeline flags / decision modes.

Writes ``artifacts/backtests/{coin}_ablation.csv`` with comparable metric columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.trade_backtest import backtests_dir, run_trade_backtest
from utils.logging_setup import get_logger

logger = get_logger(__name__)


def _slug(coin: str) -> str:
    return coin.replace(" ", "_")


def run_ablation_study(
    coin: str,
    start_date: str,
    end_date: str,
    *,
    features_dir: Path | None = None,
    models_dir: Path | None = None,
    evaluation_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """
    Run predefined variants (A–F) and save a summary CSV.

    Variants:
      A — full system
      B — no Twitter (neutral sentiment features)
      C — no regime gating (RANGING for confidence + trade filter)
      D — no cooldown simulation
      E — weaker decision-layer gates (multiplier 0.88)
      F — decision layer only (no strict ``evaluate_trade_opportunity``)
    """
    variants: list[tuple[str, dict[str, Any]]] = [
        ("A_full", {}),
        ("B_no_twitter", {"include_twitter": False}),
        ("C_no_regime_filter", {"include_regime_filter": False}),
        ("D_no_cooldown", {"include_cooldown": False}),
        ("E_weaker_thresholds", {"decision_mode": "weaker_thresholds"}),
        ("F_decision_layer_only", {"decision_mode": "decision_layer_only"}),
    ]

    rows: list[dict[str, Any]] = []
    for label, overrides in variants:
        kw = {
            "coin": coin,
            "start_date": start_date,
            "end_date": end_date,
            "features_dir": features_dir,
            "models_dir": models_dir,
            "evaluation_dir": evaluation_dir,
            **overrides,
        }
        res = run_trade_backtest(**kw)
        if res.get("error"):
            logger.warning("Ablation %s for %s: %s", label, coin, res.get("message"))
            rows.append({
                "variant": label,
                "total_trades": 0,
                "win_rate": None,
                "total_return": None,
                "max_drawdown": None,
                "sharpe_ratio": None,
                "profit_factor": None,
                "error": res.get("message", res.get("error")),
            })
            continue
        m = res.get("metrics") or {}
        rows.append({
            "variant": label,
            "total_trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate"),
            "total_return": m.get("total_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe_ratio": m.get("sharpe_ratio"),
            "profit_factor": m.get("profit_factor"),
            "error": None,
        })

    out_dir = backtests_dir()
    path = out_dir / f"{_slug(coin)}_ablation.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return rows, path


def ablation_summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
