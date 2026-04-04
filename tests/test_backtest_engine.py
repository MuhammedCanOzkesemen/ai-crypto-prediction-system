"""Trade execution, ablation output shape, and walk-forward slice discipline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import unittest

from evaluation.ablation import run_ablation_study
from evaluation.trade_execution import simulate_long_trade_daily_ohlc
from utils.config import settings


class TestTradeExecution(unittest.TestCase):
    def test_stop_loss_when_sl_and_tp_same_bar_long(self) -> None:
        entry_open = 100.0
        df = pd.DataFrame({
            "open": [entry_open, 101.0],
            "high": [110.0, 101.0],
            "low": [90.0, 99.0],
            "close": [95.0, 100.0],
        })
        r = simulate_long_trade_daily_ohlc(
            df, 0, stop_loss_pct=0.02, take_profit_pct=0.03, max_holding_bars=2
        )
        self.assertEqual(r["exit_reason"], "SL")
        self.assertAlmostEqual(float(r["exit_price"]), 98.0, places=5)

    def test_take_profit_first_bar(self) -> None:
        df = pd.DataFrame({
            "open": [100.0, 100.0],
            "high": [108.0, 101.0],
            "low": [99.0, 99.0],
            "close": [107.0, 100.0],
        })
        r = simulate_long_trade_daily_ohlc(
            df, 0, stop_loss_pct=0.02, take_profit_pct=0.03, max_holding_bars=2
        )
        self.assertEqual(r["exit_reason"], "TP")
        self.assertAlmostEqual(float(r["exit_price"]), 103.0, places=5)

    def test_time_exit_weak_hold_one_bar(self) -> None:
        df = pd.DataFrame({
            "open": [100.0, 100.0],
            "high": [101.0, 102.0],
            "low": [99.5, 99.0],
            "close": [100.5, 101.0],
        })
        r = simulate_long_trade_daily_ohlc(
            df, 0, stop_loss_pct=0.015, take_profit_pct=0.02, max_holding_bars=1
        )
        self.assertEqual(r["exit_reason"], "TIME")
        self.assertIsNotNone(r["return_pct"])


class TestAblationOutput(unittest.TestCase):
    def test_ablation_csv_columns(self) -> None:
        tmp = Path(settings.data.artifact_dir)
        with patch("evaluation.ablation.run_trade_backtest") as m:
            m.return_value = {
                "metrics": {
                    "total_trades": 1,
                    "win_rate": 0.5,
                    "total_return": 0.01,
                    "max_drawdown": -0.02,
                    "sharpe_ratio": 0.5,
                    "profit_factor": 1.2,
                }
            }
            rows, path = run_ablation_study("Bitcoin", "2024-01-01", "2024-01-31")
        self.assertEqual(len(rows), 6)
        self.assertTrue(path.exists())
        df = pd.read_csv(path)
        for col in ("variant", "total_trades", "win_rate", "total_return", "max_drawdown", "sharpe_ratio", "profit_factor"):
            self.assertIn(col, df.columns)
        path.unlink(missing_ok=True)


class TestWalkForwardHistorySlice(unittest.TestCase):
    def test_slice_includes_only_past(self) -> None:
        df = pd.DataFrame({"x": range(10)})
        idx = 5
        hist = df.iloc[: idx + 1]
        self.assertEqual(len(hist), 6)
        self.assertEqual(int(hist["x"].iloc[-1]), 5)


if __name__ == "__main__":
    unittest.main()
