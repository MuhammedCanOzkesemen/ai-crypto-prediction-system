import numpy as np
import pandas as pd
import pytest

from core.live.predictor import run_prediction


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 500
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    features = pd.DataFrame(rng.standard_normal((n, 9)), index=index)
    features.columns = [f"feature_{i}" for i in range(9)]
    features["realized_vol_5d"] = rng.uniform(0.01, 0.05, size=n)
    labels = pd.Series(rng.integers(0, 2, size=n), index=index)
    return features, labels


@pytest.fixture(scope="module")
def prediction_result(synthetic_data):
    features, labels = synthetic_data
    return run_prediction(features, labels)


def test_result_contains_required_keys(prediction_result):
    assert "probability" in prediction_result
    assert "signal" in prediction_result
    assert "reasons" in prediction_result
    assert "metrics" in prediction_result


def test_signal_is_valid(prediction_result):
    assert prediction_result["signal"] in {"BUY", "NO_TRADE"}


def test_probability_in_range(prediction_result):
    assert 0.0 <= prediction_result["probability"] <= 1.0


def test_reasons_is_list(prediction_result):
    assert isinstance(prediction_result["reasons"], list)


def test_metrics_is_dict(prediction_result):
    assert isinstance(prediction_result["metrics"], dict)
    assert len(prediction_result["metrics"]) > 0
