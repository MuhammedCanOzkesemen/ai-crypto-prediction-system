#!/usr/bin/env python3
"""
CLI for walk-forward trade backtests and optional ablation / multi-coin leaderboard.

Examples:
  python scripts/run_backtest.py --coin Bitcoin
  python scripts/run_backtest.py --coin PEPE --start 2024-01-01 --end 2025-12-31
  python scripts/run_backtest.py --all --ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from evaluation.ablation import run_ablation_study
from evaluation.trade_backtest import (
    backtests_dir,
    compare_live_and_backtest_prediction,
    run_trade_backtest,
)
from utils.config import settings
from utils.constants import SUPPORTED_COINS
from utils.logging_setup import configure_root_logger, get_logger

logger = get_logger(__name__)


def _resolve_coin(name: str) -> str | None:
    n = name.strip().replace("_", " ")
    for c in SUPPORTED_COINS:
        if c.lower() == n.lower():
            return c
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run trade backtest(s) and write artifacts under artifacts/backtests/")
    p.add_argument("--coin", type=str, default=None, help="Display name e.g. Bitcoin")
    p.add_argument("--all", action="store_true", help="Run every supported coin with data + models")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default: first valid date in parquet)")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: last date in parquet)")
    p.add_argument(
        "--compare-date",
        type=str,
        default=None,
        help="YYYY-MM-DD parity check between live-style and backtest-style prediction on that date",
    )
    p.add_argument("--ablation", action="store_true", help="Run ablation grid (A–F) for each selected coin")
    p.add_argument("--no-twitter", action="store_true")
    p.add_argument("--no-regime", action="store_true")
    p.add_argument("--no-cooldown", action="store_true")
    p.add_argument(
        "--decision-mode",
        type=str,
        default="trade_decision",
        help="trade_decision | weaker_thresholds | decision_layer_only",
    )
    return p.parse_args()


def _default_dates(coin: str) -> tuple[str, str]:
    from evaluation.trade_backtest import load_ohlcv_frame

    df = load_ohlcv_frame(coin)
    if df is None or df.empty:
        return "2000-01-01", "2099-12-31"
    d0 = pd.Timestamp(df["date"].iloc[0]).strftime("%Y-%m-%d")
    d1 = pd.Timestamp(df["date"].iloc[-1]).strftime("%Y-%m-%d")
    return d0, d1


def main() -> int:
    configure_root_logger()
    args = _parse_args()
    if not args.all and not args.coin:
        logger.error("Provide --coin NAME or --all")
        return 2

    coins: list[str]
    if args.all:
        coins = list(SUPPORTED_COINS)
    else:
        c = _resolve_coin(args.coin or "")
        if not c:
            logger.error("Unknown coin %r; supported: %s", args.coin, ", ".join(SUPPORTED_COINS))
            return 2
        coins = [c]

    leaderboard_rows: list[dict] = []
    out_dir = backtests_dir()

    for coin in coins:
        ds, de = _default_dates(coin)
        start = args.start or ds
        end = args.end or de

        if args.compare_date:
            cmp_res = compare_live_and_backtest_prediction(
                coin,
                args.compare_date,
                decision_mode=args.decision_mode,
                include_twitter=not args.no_twitter,
                include_regime_filter=not args.no_regime,
                include_cooldown=not args.no_cooldown,
            )
            logger.info("%s parity %s", coin, "OK" if cmp_res.get("matches") else "MISMATCH")
            print(json.dumps(cmp_res, indent=2, default=str))
            continue

        if args.ablation:
            rows, ab_path = run_ablation_study(coin, start, end)
            logger.info("Ablation for %s -> %s (%d variants)", coin, ab_path, len(rows))
            full_row = next((r for r in rows if r.get("variant") == "A_full"), rows[0] if rows else {})
            m_err = full_row.get("error")
            leaderboard_rows.append({
                "coin": coin,
                "total_trades": full_row.get("total_trades", 0),
                "win_rate": full_row.get("win_rate"),
                "total_return": full_row.get("total_return"),
                "max_drawdown": full_row.get("max_drawdown"),
                "sharpe_ratio": full_row.get("sharpe_ratio"),
                "profit_factor": full_row.get("profit_factor"),
                "best_regime": None,
                "notes": "ablation_run; see ablation.csv" if not m_err else str(m_err),
            })
            continue

        res = run_trade_backtest(
            coin,
            start,
            end,
            decision_mode=args.decision_mode,
            include_twitter=not args.no_twitter,
            include_regime_filter=not args.no_regime,
            include_cooldown=not args.no_cooldown,
        )
        if res.get("error"):
            logger.warning("%s: %s", coin, res.get("message"))
            leaderboard_rows.append({
                "coin": coin,
                "total_trades": 0,
                "win_rate": None,
                "total_return": None,
                "max_drawdown": None,
                "sharpe_ratio": None,
                "profit_factor": None,
                "best_regime": None,
                "notes": str(res.get("message", res.get("error"))),
            })
            continue
        m = res.get("metrics") or {}
        leaderboard_rows.append({
            "coin": coin,
            "total_trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate"),
            "total_return": m.get("total_return"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe_ratio": m.get("sharpe_ratio"),
            "profit_factor": m.get("profit_factor"),
            "best_regime": m.get("best_regime"),
            "notes": "",
        })
        logger.info(
            "%s trades=%s return=%s sharpe=%s -> %s",
            coin,
            m.get("total_trades"),
            m.get("total_return"),
            m.get("sharpe_ratio"),
            res.get("artifacts", {}).get("summary_json"),
        )

    if args.all:
        lb_path = out_dir / "backtest_leaderboard.csv"
        pd.DataFrame(leaderboard_rows).to_csv(lb_path, index=False)
        logger.info("Leaderboard -> %s", lb_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
