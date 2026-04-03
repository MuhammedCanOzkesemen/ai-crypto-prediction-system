"""Unit tests for trade decision layer (no model I/O)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction.decision_layer import (
    categorize_expected_move_strength,
    compute_decision_bundle,
    compute_edge_score,
    compute_risk_reward_ratio,
    expected_move_pct,
)


class TestDecisionLayer(unittest.TestCase):
    def test_expected_move_pct_and_strength(self) -> None:
        self.assertAlmostEqual(expected_move_pct(100.0, 103.0), 0.03)
        self.assertEqual(categorize_expected_move_strength(0.01), "WEAK")
        self.assertEqual(categorize_expected_move_strength(0.03), "MODERATE")
        self.assertEqual(categorize_expected_move_strength(0.06), "STRONG")

    def test_risk_reward_ratio(self) -> None:
        # spot 100, final 110, band 95-105 -> reward 10, risk 10 -> 1.0
        rr = compute_risk_reward_ratio(100.0, 110.0, 95.0, 105.0)
        self.assertAlmostEqual(rr, 1.0, places=5)
        # inverted bounds handled
        rr2 = compute_risk_reward_ratio(100.0, 110.0, 105.0, 95.0)
        self.assertAlmostEqual(rr2, 1.0, places=5)

    def test_edge_score_in_zero_one(self) -> None:
        e = compute_edge_score(
            0.04,
            0.5,
            {"up": 0.55, "down": 0.2, "neutral": 0.25},
            {"adx_14": 30.0},
        )
        self.assertTrue(0.0 <= e <= 1.0)

    def test_default_no_trade_when_weak_move(self) -> None:
        d = compute_decision_bundle(
            current_price=50000.0,
            final_prediction=50100.0,  # 0.2% move
            lower_bound=49000.0,
            upper_bound=51000.0,
            base_confidence=0.9,
            mean_path_agreement=0.95,
            signal_strength_score=0.9,
            trend_label="UP",
            directional_probs={"up": 0.7, "down": 0.15, "neutral": 0.15},
            volatility_regime="LOW",
            feature_context={"adx_14": 35.0},
        )
        self.assertEqual(d["trade_signal"], "NO_TRADE")
        self.assertFalse(d["trade_valid"])
        self.assertEqual(d["expected_move_strength"], "WEAK")

    def test_no_trade_when_neutral_trend(self) -> None:
        d = compute_decision_bundle(
            current_price=100.0,
            final_prediction=108.0,
            lower_bound=92.0,
            upper_bound=112.0,
            base_confidence=0.85,
            mean_path_agreement=0.85,
            signal_strength_score=0.7,
            trend_label="NEUTRAL",
            directional_probs={"up": 0.6, "down": 0.2, "neutral": 0.2},
            volatility_regime="MEDIUM",
            feature_context={"adx_14": 28.0},
        )
        self.assertEqual(d["trade_signal"], "NO_TRADE")

    def test_buy_when_strong_aligned_setup(self) -> None:
        d = compute_decision_bundle(
            current_price=100.0,
            final_prediction=120.0,
            lower_bound=98.0,
            upper_bound=105.0,
            base_confidence=0.72,
            mean_path_agreement=0.75,
            signal_strength_score=0.72,
            trend_label="STRONG UP",
            directional_probs={"up": 0.62, "down": 0.18, "neutral": 0.2},
            volatility_regime="LOW",
            feature_context={"adx_14": 32.0},
        )
        self.assertEqual(d["trade_signal"], "BUY")
        self.assertTrue(d["trade_valid"])
        self.assertTrue(d["directional_alignment"])


if __name__ == "__main__":
    unittest.main()
