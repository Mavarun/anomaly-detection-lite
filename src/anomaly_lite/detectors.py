"""Unsupervised anomaly detectors wrapping scikit-learn estimators."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
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


class LOFDetector:
    """Local Outlier Factor with optional standardisation.

    ``novelty=False`` (default) is the transductive mode: the detector
    scores exactly the rows it was fitted on (each point's own neighbours
    exclude itself). Scoring other rows then raises -- use ``novelty=True``
    to fit on a reference set and score new data. LOF is distance based, so
    features are standardised by default.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float | str = 0.1,
        novelty: bool = False,
        scale: bool = True,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty
        self.scale = scale
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination, novelty=novelty
        )
        self._scaler: StandardScaler | None = None
        self._X_fit: np.ndarray | None = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self._scaler.transform(X) if self._scaler is not None else X

    def fit(self, X: np.ndarray) -> LOFDetector:
        X = np.asarray(X, dtype=float)
        self._scaler = StandardScaler().fit(X) if self.scale else None
        self.model.fit(self._prep(X))
        self._X_fit = X.copy()
        return self

    def _check_transductive(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        if self._X_fit is None:
            raise RuntimeError("LOFDetector is not fitted")
        if X.shape != self._X_fit.shape or not np.array_equal(X, self._X_fit):
            raise ValueError(
                "transductive LOF can only score its fit data; use novelty=True"
            )

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.novelty:
            return -self.model.score_samples(self._prep(X))
        self._check_transductive(X)
        return -self.model.negative_outlier_factor_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.novelty:
            return (self.model.predict(self._prep(X)) == -1).astype(int)
        self._check_transductive(X)
        thr = self.model.offset_
        return (self.model.negative_outlier_factor_ < thr).astype(int)
