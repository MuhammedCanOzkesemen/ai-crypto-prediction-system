import numpy as np
import pandas as pd
import pytest

from core.models.trainer import walk_forward_train


@pytest.fixture(scope="module")
def synthetic_dataset():
    rng = np.random.default_rng(42)
    n = 500
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    features = pd.DataFrame(rng.standard_normal((n, 10)), index=index)
    labels = pd.Series(rng.integers(0, 2, size=n), index=index)
    return features, labels


@pytest.fixture(scope="module")
def fold_results(synthetic_dataset):
    features, labels = synthetic_dataset
    return walk_forward_train(features, labels, n_splits=5, random_seed=42)


def test_fold_count(fold_results):
    assert len(fold_results) > 1


def test_no_data_leakage(synthetic_dataset, fold_results):
    features, labels = synthetic_dataset
    n = len(features)
    min_train = int(n * 0.5)
    fold_size = (n - min_train) // len(fold_results)

    for result in fold_results:
        train_end = min_train + result.fold * fold_size
        val_start = train_end
        assert train_end <= val_start


def test_prediction_lengths_match_validation(synthetic_dataset, fold_results):
    features, labels = synthetic_dataset
    n = len(features)
    min_train = int(n * 0.5)
    n_splits = len(fold_results)
    fold_size = (n - min_train) // n_splits

    for result in fold_results:
        val_start = min_train + result.fold * fold_size
        val_end = val_start + fold_size if result.fold < n_splits - 1 else n
        expected_size = val_end - val_start
        assert len(result.y_pred) == expected_size
        assert len(result.y_prob) == expected_size
        assert len(result.y_true) == expected_size


def test_probabilities_in_range(fold_results):
    for result in fold_results:
        assert np.all(result.y_prob >= 0.0)
        assert np.all(result.y_prob <= 1.0)
