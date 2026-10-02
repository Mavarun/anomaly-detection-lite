"""Tests for LOF and the IsolationForest-vs-LOF tabular benchmark."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from anomaly_lite.benchmark import (
    load_breast_cancer_anomaly,
    make_local_density_anomalies,
    paired_wins,
    run_tabular_benchmark,
    summarise_benchmark,
)
from anomaly_lite.data import make_contaminated_gaussian
from anomaly_lite.detectors import IsolationForestDetector, LOFDetector


def test_lof_transductive_refuses_new_rows_and_novelty_scores_them():
    X, _ = make_local_density_anomalies(random_state=0)
    det = LOFDetector().fit(X)
    s = det.score_samples(X)
    assert s.shape == (len(X),) and np.all(s > 0)  # LOF >= ~1 for inliers
    with pytest.raises(ValueError):
        det.score_samples(X[:10])
    nov = LOFDetector(novelty=True).fit(X)
    far = np.full((1, X.shape[1]), 50.0)
    assert nov.score_samples(far)[0] > np.quantile(nov.score_samples(X[:50]), 0.99)
    assert det.predict(X).sum() > 0


def test_local_density_dataset_lof_beats_isolation_forest():
    for seed in range(3):
        X, y = make_local_density_anomalies(random_state=seed)
        ap_lof = average_precision_score(y, LOFDetector(n_neighbors=20).fit(X).score_samples(X))
        ap_if = average_precision_score(
            y, IsolationForestDetector(contamination=0.05, random_state=seed).fit(X).score_samples(X)
        )
        assert ap_lof > ap_if + 0.2


def test_lof_masking_when_anomalies_cluster_and_k_is_small():
    X, y = make_contaminated_gaussian(n_normal=900, n_anomaly=100, n_features=8, random_state=0)
    X = X.to_numpy()
    auc_k20 = roc_auc_score(y, LOFDetector(n_neighbors=20).fit(X).score_samples(X))
    auc_k150 = roc_auc_score(y, LOFDetector(n_neighbors=150).fit(X).score_samples(X))
    assert auc_k20 < 0.6  # 100 clustered anomalies look like a normal cluster to k=20
    assert auc_k150 > 0.95


def test_breast_cancer_anomaly_loader():
    X, y = load_breast_cancer_anomaly(n_anomalies=21, random_state=0)
    assert X.shape == (378, 30) and y.sum() == 21
    X2, y2 = load_breast_cancer_anomaly(n_anomalies=21, random_state=0)
    np.testing.assert_array_equal(X, X2)
    X3, y3 = load_breast_cancer_anomaly(n_anomalies=21, random_state=1)
    assert not np.array_equal(np.sort(X[y == 1], axis=0), np.sort(X3[y3 == 1], axis=0))
    with pytest.raises(ValueError):
        load_breast_cancer_anomaly(n_anomalies=0)


def test_benchmark_frame_and_summary():
    long = run_tabular_benchmark(datasets=("local_density", "breast_cancer"), seeds=range(2))
    assert len(long) == 2 * 2 * 4
    for col in ("roc_auc", "avg_precision", "precision_at_n_anom"):
        assert long[col].between(0, 1).all()
    summ = summarise_benchmark(long)
    assert {"roc_auc_mean", "avg_precision_std"} <= set(summ.columns)
    wins = paired_wins(long, "lof_k20", "isolation_forest")
    assert wins.loc["local_density"] == 1.0
