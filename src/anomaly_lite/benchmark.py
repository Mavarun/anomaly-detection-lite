"""IsolationForest vs LOF (vs OneClassSVM) on labelled tabular anomaly sets.

Datasets (all offline):

* ``global_offset`` -- the repo's original contaminated Gaussian
  (anomalies shifted far from one inlier blob): an easy *global* outlier task.
* ``local_density`` -- two inlier clusters of very different density
  (tight N(0, 0.3^2) and diffuse N(6, 1.5^2)) with anomalies placed in a
  thin shell 1.2-2.0 units from the tight cluster's centre. Anomalies are
  closer to dense data than many diffuse-cluster inliers are to each other,
  so a global isolation score is expected to struggle and LOF to help.
* ``breast_cancer`` -- public sklearn Wisconsin breast-cancer data in the
  ODDS-style anomaly setup: all 357 benign rows are inliers and a seeded
  random subset of ``n_anomalies`` (default 21) malignant rows are anomalies.
  Seed-to-seed variance therefore includes *which* malignant rows are drawn.

All detectors are fitted transductively on the full unlabeled matrix (no
labels used); labels only enter the metrics: ROC-AUC, average precision,
and precision@n_anomalies.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import average_precision_score, roc_auc_score

from anomaly_lite.data import make_contaminated_gaussian
from anomaly_lite.detectors import (
    IsolationForestDetector,
    LOFDetector,
    OneClassSVMDetector,
)
from anomaly_lite.metrics import precision_at_k


def make_local_density_anomalies(
    n_dense: int = 400,
    n_diffuse: int = 400,
    n_anomaly: int = 40,
    n_features: int = 2,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-density inliers plus a shell of local anomalies near the dense blob."""
    rng = np.random.default_rng(random_state)
    dense = rng.normal(0.0, 0.3, size=(n_dense, n_features))
    diffuse = rng.normal(6.0, 1.5, size=(n_diffuse, n_features))
    direction = rng.normal(size=(n_anomaly, n_features))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = rng.uniform(1.2, 2.0, size=(n_anomaly, 1))
    anom = direction * radius
    X = np.vstack([dense, diffuse, anom])
    y = np.r_[np.zeros(n_dense + n_diffuse, dtype=int), np.ones(n_anomaly, dtype=int)]
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_breast_cancer_anomaly(
    n_anomalies: int = 21, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Benign rows as inliers + a seeded subsample of malignant rows as anomalies."""
    bunch = load_breast_cancer()
    X, target = np.asarray(bunch.data, dtype=float), np.asarray(bunch.target)
    benign, malignant = X[target == 1], X[target == 0]  # sklearn: 0 = malignant
    if not 1 <= n_anomalies <= len(malignant):
        raise ValueError("n_anomalies out of range")
    rng = np.random.default_rng(random_state)
    pick = malignant[rng.choice(len(malignant), size=n_anomalies, replace=False)]
    Xa = np.vstack([benign, pick])
    y = np.r_[np.zeros(len(benign), dtype=int), np.ones(n_anomalies, dtype=int)]
    perm = rng.permutation(len(y))
    return Xa[perm], y[perm]


def _global_offset(seed: int):
    X, y = make_contaminated_gaussian(n_normal=900, n_anomaly=100, n_features=8,
                                      random_state=seed)
    return X.to_numpy(), y


DATASETS: dict[str, Callable[[int], tuple[np.ndarray, np.ndarray]]] = {
    "global_offset": _global_offset,
    "local_density": lambda seed: make_local_density_anomalies(random_state=seed),
    "breast_cancer": lambda seed: load_breast_cancer_anomaly(random_state=seed),
}


def build_detectors(seed: int, contamination: float) -> dict:
    return {
        "isolation_forest": IsolationForestDetector(contamination=contamination,
                                                    random_state=seed),
        "lof_k20": LOFDetector(n_neighbors=20, contamination=contamination),
        # k larger than the planted anomaly count in global_offset (100):
        # probes LOF's masking failure when anomalies form their own cluster
        "lof_k150": LOFDetector(n_neighbors=150, contamination=contamination),
        "ocsvm": OneClassSVMDetector(nu=contamination),
    }


def run_tabular_benchmark(
    datasets: Iterable[str] = tuple(DATASETS),
    seeds: Iterable[int] = range(10),
) -> pd.DataFrame:
    """Long frame: one row per (dataset, seed, detector).

    ``contamination`` / ``nu`` is set to the true prevalence (an optimistic
    choice that only affects the binary ``predict`` columns, not the
    ranking metrics). OCSVM gets standardised inputs; IF does not need them.
    """
    rows = []
    for name in datasets:
        loader = DATASETS[name]
        for seed in seeds:
            X, y = loader(seed)
            prev = float(y.mean())
            k = int(y.sum())
            for det_name, det in build_detectors(seed, prev).items():
                Xin = X
                if det_name == "ocsvm":
                    Xin = (X - X.mean(0)) / X.std(0).clip(min=1e-12)
                det.fit(Xin)
                s = det.score_samples(Xin)
                rows.append({
                    "dataset": name, "seed": seed, "detector": det_name,
                    "n": int(len(y)), "prevalence": prev,
                    "roc_auc": float(roc_auc_score(y, s)),
                    "avg_precision": float(average_precision_score(y, s)),
                    "precision_at_n_anom": float(precision_at_k(y, s, k)),
                })
    return pd.DataFrame(rows)


def summarise_benchmark(long: pd.DataFrame) -> pd.DataFrame:
    metrics = ["roc_auc", "avg_precision", "precision_at_n_anom"]
    agg = long.groupby(["dataset", "detector"], sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def paired_wins(long: pd.DataFrame, a: str, b: str, metric: str = "avg_precision") -> pd.Series:
    """Per dataset, share of seeds where detector ``a`` beats ``b`` on ``metric``."""
    piv = long.pivot_table(index=["dataset", "seed"], columns="detector", values=metric)
    return (piv[a] > piv[b]).groupby(level="dataset").mean().rename(f"{a}_beats_{b}")
