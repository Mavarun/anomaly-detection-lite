"""Synthetic contaminated Gaussian data for anomaly slice demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_contaminated_gaussian(
    n_normal: int = 900,
    n_anomaly: int = 100,
    n_features: int = 8,
    normal_scale: float = 1.0,
    anomaly_offset: float = 4.0,
    anomaly_scale: float = 1.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate inliers ~ N(0, s) and anomalies ~ N(offset, s_a).

    Returns feature frame and binary labels (1 = anomaly).
    """
    rng = np.random.default_rng(random_state)
    X_normal = rng.normal(0.0, normal_scale, size=(n_normal, n_features))
    X_anom = rng.normal(
        anomaly_offset, anomaly_scale, size=(n_anomaly, n_features)
    )
    X = np.vstack([X_normal, X_anom])
    y = np.concatenate(
        [np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)]
    )
    # shuffle jointly
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y
