"""Strict trade filter unit tests."""

from __future__ import annotations

import unittest

from prediction.trade_filter import compute_trade_engine_edge_score, evaluate_trade_opportunity


class TestTradeFilter(unittest.TestCase):
    def test_edge_product(self) -> None:
        e = compute_trade_engine_edge_score(0.5, 0.8, 0.5)
        self.assertAlmostEqual(e, 0.2, places=5)

    def test_hard_filter(self) -> None:
        r = evaluate_trade_opportunity(
            {
                "confidence_score": 0.15,
                "mean_path_agreement": 0.9,
                "signal_strength_score": 0.9,
                "directional_probabilities": {"up": 0.9, "down": 0.05, "neutral": 0.05},
                "directional_confidence": 0.9,
                "combined_agreement_score": 0.7,
                "volatility_regime": "LOW",
                "expected_move_pct": 0.1,
                "risk_reward_ratio": 2.0,
                "trend_label": "UP",
                "forecast_bullish": True,
            }
        )
        self.assertEqual(r["decision"], "NO_TRADE")
        self.assertTrue(any("Hard filter" in x for x in r["reasons"]))

    def test_strong_buy(self) -> None:
        r = evaluate_trade_opportunity(
            {
                "confidence_score": 0.58,
                "mean_path_agreement": 0.65,
                "signal_strength_score": 0.55,
                "directional_probabilities": {"up": 0.65, "down": 0.2, "neutral": 0.15},
                "directional_confidence": 0.58,
                "combined_agreement_score": 0.62,
                "volatility_regime": "MEDIUM",
                "expected_move_pct": 0.04,
                "risk_reward_ratio": 1.6,
                "trend_label": "UP",
                "forecast_bullish": True,
                "adx_14": 28.0,
                "sentiment_alignment": 1.0,
                "consensus_score": 0.75,
                "stability_score": 0.65,
                "trend_confirmation_score": 0.6,
                "volatility_shock_detected": False,
                "market_regime": "TRENDING",
            }
        )
        self.assertEqual(r["decision"], "STRONG_BUY")
        self.assertGreater(r["score"], 0.0)

    def test_ranging_blocks_strong(self) -> None:
        r = evaluate_trade_opportunity(
            {
                "confidence_score": 0.58,
                "mean_path_agreement": 0.65,
                "signal_strength_score": 0.55,
                "directional_probabilities": {"up": 0.65, "down": 0.2, "neutral": 0.15},
                "directional_confidence": 0.58,
                "combined_agreement_score": 0.62,
                "volatility_regime": "MEDIUM",
                "expected_move_pct": 0.04,
                "risk_reward_ratio": 1.6,
                "trend_label": "UP",
                "forecast_bullish": True,
                "consensus_score": 0.75,
                "stability_score": 0.65,
                "trend_confirmation_score": 0.6,
                "volatility_shock_detected": False,
                "market_regime": "RANGING",
            }
        )
        self.assertNotEqual(r["decision"], "STRONG_BUY")

    def test_strong_downgrade_shock(self) -> None:
        r = evaluate_trade_opportunity(
            {
                "confidence_score": 0.58,
                "mean_path_agreement": 0.65,
                "signal_strength_score": 0.55,
                "directional_probabilities": {"up": 0.65, "down": 0.2, "neutral": 0.15},
                "directional_confidence": 0.58,
                "combined_agreement_score": 0.62,
                "volatility_regime": "MEDIUM",
                "expected_move_pct": 0.04,
                "risk_reward_ratio": 1.6,
                "trend_label": "UP",
                "forecast_bullish": True,
                "consensus_score": 0.75,
                "stability_score": 0.65,
                "trend_confirmation_score": 0.6,
                "volatility_shock_detected": True,
                "market_regime": "TRENDING",
            }
        )
        self.assertNotEqual(r["decision"], "STRONG_BUY")


if __name__ == "__main__":
    unittest.main()
