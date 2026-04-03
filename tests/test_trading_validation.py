"""Trading validation helpers."""

from __future__ import annotations

import unittest

from prediction.trading_validation import (
    compose_risk_adjusted_confidence,
    lr_matrix_chaotic_disagreement,
)


class TestTradingValidation(unittest.TestCase):
    def test_chaotic_opposite_signs(self) -> None:
        m = [[0.02, -0.02, 0.02, -0.02] for _ in range(5)]
        ch, frac = lr_matrix_chaotic_disagreement(m)
        self.assertTrue(ch)
        self.assertGreaterEqual(frac, 0.28)

    def test_compose_caps(self) -> None:
        c = compose_risk_adjusted_confidence(
            0.9,
            0.8,
            0.9,
            0.9,
            0.9,
            volatility_regime_high=False,
            volatility_shock=False,
            chaotic_disagreement=True,
        )
        self.assertLessEqual(c, 0.5)


if __name__ == "__main__":
    unittest.main()
