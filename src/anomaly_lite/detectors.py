"""Unsupervised anomaly detectors wrapping scikit-learn estimators."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class AnomalyDetector(Protocol):
    def fit(self, X: np.ndarray) -> AnomalyDetector: ...

    def score_samples(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class IsolationForestDetector:
    """IsolationForest with explicit contamination handling.

    Higher anomaly_score means more anomalous (negated decision_function).
    """

    def __init__(
        self,
        contamination: float | str = 0.1,
        n_estimators: int = 200,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray) -> IsolationForestDetector:
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # sklearn: lower score_samples => more anomalous; flip for ranking
        return -self.model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn: -1 anomaly, 1 inlier -> 1 anomaly, 0 inlier
        return (self.model.predict(X) == -1).astype(int)


class OneClassSVMDetector:
    """OneClassSVM with nu as contamination-like prior."""

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        **kwargs,
    ) -> None:
        self.nu = nu
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, **kwargs)

    def fit(self, X: np.ndarray) -> OneClassSVMDetector:
        self.model.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(X) == -1).astype(int)
