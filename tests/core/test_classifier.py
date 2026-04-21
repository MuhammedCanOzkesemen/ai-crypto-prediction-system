import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from core.models.classifier import CryptoClassifier


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


@pytest.fixture(scope="module")
def fitted_classifier(dataset):
    X_train, _, _, y_train, _, _ = dataset
    clf = CryptoClassifier(random_seed=42)
    clf.fit(X_train, y_train)
    return clf


def test_fit_runs_without_error(dataset):
    X_train, _, _, y_train, _, _ = dataset
    clf = CryptoClassifier(random_seed=42)
    clf.fit(X_train, y_train)


def test_predict_proba_range(fitted_classifier, dataset):
    _, _, X_test, _, _, _ = dataset
    proba = fitted_classifier.predict_proba(X_test)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_predict_proba_shape(fitted_classifier, dataset):
    _, _, X_test, _, _, _ = dataset
    proba = fitted_classifier.predict_proba(X_test)
    assert proba.shape == (X_test.shape[0], 2)


def test_calibrate_improves_brier_score(dataset):
    X_train, X_val, X_test, y_train, y_val, y_test = dataset
    clf = CryptoClassifier(random_seed=42)
    clf.fit(X_train, y_train)

    proba_before = clf.predict_proba(X_test)[:, 1]
    brier_before = brier_score_loss(y_test, proba_before)

    clf.calibrate(X_val, y_val)

    proba_after = clf.predict_proba(X_test)[:, 1]
    brier_after = brier_score_loss(y_test, proba_after)

    assert brier_after <= brier_before
