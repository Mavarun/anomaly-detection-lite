"""Ranking metrics for anomaly detection evaluation."""

from __future__ import annotations

import numpy as np


def precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> float:
    """Precision@k: fraction of true anomalies among top-k scored points.

    Args:
        y_true: Binary labels (1 = anomaly).
        scores: Anomaly scores (higher = more anomalous).
        k: Number of top-ranked predictions to evaluate.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if k <= 0:
        raise ValueError("k must be positive")
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    k = min(k, len(scores))
    top_idx = np.argsort(-scores)[:k]
    return float(y_true[top_idx].sum() / k)


def random_precision_at_k(y_true: np.ndarray, k: int) -> float:
    """Expected precision@k under uniform random ranking = anomaly rate."""
    y_true = np.asarray(y_true).astype(int)
    if k <= 0:
        raise ValueError("k must be positive")
    return float(y_true.mean())
