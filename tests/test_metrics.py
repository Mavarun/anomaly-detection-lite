"""Tests for precision@k and random baseline."""

import numpy as np
import pytest

from anomaly_lite.data import make_contaminated_gaussian
from anomaly_lite.detectors import IsolationForestDetector
from anomaly_lite.metrics import precision_at_k, random_precision_at_k


def test_precision_at_k_perfect():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([5.0, 4.0, 1.0, 0.5, 0.0])
    assert precision_at_k(y, scores, k=2) == 1.0
    assert precision_at_k(y, scores, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_beats_random_on_synthetic():
    X, y = make_contaminated_gaussian(
        n_normal=500, n_anomaly=50, n_features=8, random_state=7
    )
    det = IsolationForestDetector(contamination=0.1, random_state=7).fit(X.values)
    scores = det.score_samples(X.values)
    k = 50
    p_at_k = precision_at_k(y, scores, k=k)
    baseline = random_precision_at_k(y, k=k)
    assert p_at_k > baseline
    assert p_at_k >= 0.5  # should recover most of top-k as true anomalies


def test_random_precision_equals_prevalence():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    assert random_precision_at_k(y, k=3) == pytest.approx(0.2)


def test_precision_at_k_rejects_bad_k():
    with pytest.raises(ValueError):
        precision_at_k(np.array([1]), np.array([0.5]), k=0)
