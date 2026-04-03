"""Tests for directional vs chaotic model disagreement handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction.forecast_intel import mean_effective_logret_variance


class TestDisagreement(unittest.TestCase):
    def test_same_sign_lower_effective_var(self) -> None:
        eff, raw = mean_effective_logret_variance([[0.01, 0.02, 0.015]])
        self.assertGreater(raw, 0)
        self.assertLess(eff, raw)

    def test_mixed_sign_full_var(self) -> None:
        eff, raw = mean_effective_logret_variance([[0.02, -0.02, 0.01]])
        self.assertAlmostEqual(eff, raw, places=8)


if __name__ == "__main__":
    unittest.main()
