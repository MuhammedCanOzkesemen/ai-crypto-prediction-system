"""
Contract tests: path-quality signals and recursive diagnostics must never omit required keys.
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction.forecast_path_quality import (
    compute_path_output_signals,
    normalize_forecast_path_signals,
)
from prediction.recursive_forecast import _normalize_recursive_forecast_diagnostics
from prediction.forecast_validity import compute_forecast_validity_and_quality_score


REQUIRED_PATH_SIGNAL_KEYS = frozenset(
    {
        "is_constant_prediction",
        "low_variance_warning",
        "degraded_input",
        "ensemble_log_return_var",
        "price_relative_variance",
        "model_horizon_variance_raw",
        "step0_raw_logret_by_model",
        "max_missing_feature_fill_ratio",
    }
)


class TestPathOutputSignals(unittest.TestCase):
    def test_empty_path_includes_max_missing_feature_fill_ratio(self) -> None:
        raw = compute_path_output_signals(
            [],
            {},
            max_missing_feature_fill_ratio=0.4,
            emergency_width_align_used=True,
            spot_start=50000.0,
        )
        self.assertIn("max_missing_feature_fill_ratio", raw)
        self.assertEqual(raw["max_missing_feature_fill_ratio"], 0.4)
        for k in REQUIRED_PATH_SIGNAL_KEYS:
            self.assertIn(k, raw, msg=f"missing {k} on empty path")

    def test_normalize_fills_partial_dict_like_old_bug(self) -> None:
        partial = {
            "is_constant_prediction": True,
            "low_variance_warning": True,
            "degraded_input": True,
            "ensemble_log_return_var": 0.0,
            "price_relative_variance": 0.0,
            "model_horizon_variance_raw": {},
            "step0_raw_logret_by_model": {},
        }
        n = normalize_forecast_path_signals(partial)
        self.assertEqual(n["max_missing_feature_fill_ratio"], 0.0)
        for k in REQUIRED_PATH_SIGNAL_KEYS:
            self.assertIn(k, n)

    def test_normalize_none_returns_full_defaults(self) -> None:
        n = normalize_forecast_path_signals(None)
        for k in REQUIRED_PATH_SIGNAL_KEYS:
            self.assertIn(k, n)
        self.assertEqual(n["max_missing_feature_fill_ratio"], 0.0)
        self.assertIsInstance(n["model_horizon_variance_raw"], dict)
        self.assertIsInstance(n["step0_raw_logret_by_model"], dict)


class TestForecastValiditySparseDiag(unittest.TestCase):
    def test_empty_diag_no_exception(self) -> None:
        v, q = compute_forecast_validity_and_quality_score({})
        self.assertIn(v, ("invalid", "questionable", "valid"))
        self.assertIsInstance(q, float)

    def test_production_artifact_sparse(self) -> None:
        v, q = compute_forecast_validity_and_quality_score({"artifact_mode": "production"})
        self.assertEqual(v, "valid")
        self.assertGreater(q, 0.5)

    def test_legacy_artifact_sparse(self) -> None:
        v, q = compute_forecast_validity_and_quality_score({"artifact_mode": "legacy"})
        self.assertIn(v, ("invalid", "questionable", "valid"))
        self.assertLess(q, 0.5)

    def test_degraded_input_flag_sparse(self) -> None:
        v, q = compute_forecast_validity_and_quality_score(
            {"artifact_mode": "production", "degraded_input": True}
        )
        self.assertEqual(v, "questionable")


class TestRecursiveDiagnosticsNormalization(unittest.TestCase):
    def test_partial_diagnostics_get_defaults(self) -> None:
        minimal = {"forecast_quality": "production", "used_scaler": True}
        out = _normalize_recursive_forecast_diagnostics(minimal)
        self.assertEqual(out["forecast_quality"], "production")
        self.assertTrue(out["used_scaler"])
        self.assertEqual(out["max_missing_feature_fill_ratio"], 0.0)
        self.assertIn("is_constant_prediction", out)
        self.assertIn("sanity_check", out)
        self.assertIsInstance(out["sanity_check"], dict)

    def test_legacy_mode_partial_no_keyerror(self) -> None:
        out = _normalize_recursive_forecast_diagnostics({})
        self.assertEqual(out["forecast_quality"], "degraded")
        self.assertTrue(out["fallback_mode"])


if __name__ == "__main__":
    unittest.main()
