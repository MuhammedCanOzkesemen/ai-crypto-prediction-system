import numpy as np
import pytest

from core.models.trainer import FoldResult
from core.models.evaluator import evaluate_folds


EXPECTED_KEYS = {
    "accuracy_mean",
    "accuracy_std",
    "brier_score_mean",
    "brier_score_std",
    "roc_auc_mean",
    "roc_auc_std",
}


@pytest.fixture(scope="module")
def fake_fold_results():
    rng = np.random.default_rng(0)
    results = []
    for i in range(4):
        y_true = rng.integers(0, 2, size=100)
        y_prob = rng.uniform(0.0, 1.0, size=100)
        y_pred = (y_prob >= 0.5).astype(int)
        results.append(FoldResult(fold=i, y_true=y_true, y_pred=y_pred, y_prob=y_prob))
    return results


@pytest.fixture(scope="module")
def metrics(fake_fold_results):
    return evaluate_folds(fake_fold_results)


def test_all_keys_present(metrics):
    assert EXPECTED_KEYS == set(metrics.keys())


def test_accuracy_mean_in_range(metrics):
    assert 0.0 <= metrics["accuracy_mean"] <= 1.0


def test_brier_score_mean_in_range(metrics):
    assert 0.0 <= metrics["brier_score_mean"] <= 1.0


def test_roc_auc_mean_in_range(metrics):
    assert 0.0 <= metrics["roc_auc_mean"] <= 1.0


def test_std_values_non_negative(metrics):
    assert metrics["accuracy_std"] >= 0.0
    assert metrics["brier_score_std"] >= 0.0
    assert metrics["roc_auc_std"] >= 0.0
