"""Tests for IsolationForest and OneClassSVM detectors."""

import numpy as np
import pytest

from anomaly_lite.data import make_contaminated_gaussian
from anomaly_lite.detectors import IsolationForestDetector, OneClassSVMDetector


@pytest.fixture
def synthetic():
    X, y = make_contaminated_gaussian(
        n_normal=400, n_anomaly=40, n_features=6, random_state=0
    )
    return X.values, y


def test_isolation_forest_flags_anomalies(synthetic):
    X, y = synthetic
    det = IsolationForestDetector(contamination=0.1, random_state=0).fit(X)
    pred = det.predict(X)
    # Should catch a non-trivial share of planted anomalies
    recall = pred[y == 1].mean()
    assert recall >= 0.4
    assert pred.sum() > 0


def test_one_class_svm_flags_anomalies(synthetic):
    X, y = synthetic
    det = OneClassSVMDetector(nu=0.1).fit(X)
    pred = det.predict(X)
    recall = pred[y == 1].mean()
    assert recall >= 0.3
    assert pred.sum() > 0


def test_scores_rank_anomalies_higher(synthetic):
    X, y = synthetic
    det = IsolationForestDetector(contamination=0.1, random_state=0).fit(X)
    scores = det.score_samples(X)
    assert scores[y == 1].mean() > scores[y == 0].mean()
