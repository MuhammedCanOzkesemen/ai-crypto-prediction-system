"""Market regime classifier."""

from __future__ import annotations

import unittest

import pandas as pd

from prediction.market_regime import REGIME_RANGING, REGIME_TRENDING, REGIME_VOLATILE, detect_market_regime


class TestMarketRegime(unittest.TestCase):
    def test_trending_high_adx(self) -> None:
        row = pd.Series(
            {
                "adx_14": 32.0,
                "ewma_vol_logret_14": 0.02,
                "trend_consistency_score": 0.62,
                "range_position_14d_window": 0.55,
            }
        )
        out = detect_market_regime(row)
        self.assertEqual(out["regime"], REGIME_TRENDING)
        self.assertGreater(out["market_regime_confidence"], 0.4)

    def test_volatile_spike(self) -> None:
        n = 20
        vols = [0.015] * (n - 1) + [0.045]
        df = pd.DataFrame(
            {
                "adx_14": [18.0] * n,
                "ewma_vol_logret_14": vols,
                "trend_consistency_score": [0.5] * n,
                "range_position_14d_window": [0.5] * n,
            }
        )
        out = detect_market_regime(df)
        self.assertEqual(out["regime"], REGIME_VOLATILE)

    def test_ranging_low_adx(self) -> None:
        row = pd.Series(
            {
                "adx_14": 16.0,
                "ewma_vol_logret_14": 0.018,
                "trend_consistency_score": 0.42,
                "range_position_14d_window": 0.48,
            }
        )
        out = detect_market_regime(row)
        self.assertEqual(out["regime"], REGIME_RANGING)


if __name__ == "__main__":
    unittest.main()
