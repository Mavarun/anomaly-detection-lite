"""anomaly-lite: IsolationForest / OneClassSVM slice with precision@k."""

from anomaly_lite.detectors import IsolationForestDetector, OneClassSVMDetector
from anomaly_lite.metrics import precision_at_k, random_precision_at_k

__all__ = [
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "precision_at_k",
    "random_precision_at_k",
]

__version__ = "0.1.0"
