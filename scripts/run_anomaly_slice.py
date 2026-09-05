#!/usr/bin/env python3
"""Run IsolationForest / OneClassSVM on synthetic data; report precision@k."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_lite.data import make_contaminated_gaussian
from anomaly_lite.detectors import IsolationForestDetector, OneClassSVMDetector
from anomaly_lite.metrics import precision_at_k, random_precision_at_k


def evaluate(contamination: float, nu: float, k: int, seed: int = 42) -> dict:
    X, y = make_contaminated_gaussian(
        n_normal=900,
        n_anomaly=100,
        n_features=8,
        random_state=seed,
    )
    prevalence = float(y.mean())
    baseline = random_precision_at_k(y, k=k)

    iforest = IsolationForestDetector(
        contamination=contamination, random_state=seed
    ).fit(X.values)
    ocsvm = OneClassSVMDetector(nu=nu).fit(X.values)

    if_scores = iforest.score_samples(X.values)
    oc_scores = ocsvm.score_samples(X.values)

    result = {
        "n_samples": int(len(y)),
        "n_anomalies": int(y.sum()),
        "prevalence": round(prevalence, 4),
        "k": k,
        "random_precision_at_k": round(baseline, 4),
        "isolation_forest": {
            "contamination": contamination,
            "precision_at_k": round(precision_at_k(y, if_scores, k), 4),
            "flagged": int(iforest.predict(X.values).sum()),
        },
        "one_class_svm": {
            "nu": nu,
            "precision_at_k": round(precision_at_k(y, oc_scores, k), 4),
            "flagged": int(ocsvm.predict(X.values).sum()),
        },
    }
    return result


def contamination_sensitivity(k: int = 100, seed: int = 42) -> list[dict]:
    rows = []
    for c in (0.05, 0.10, 0.15, 0.20):
        r = evaluate(contamination=c, nu=c, k=k, seed=seed)
        rows.append(
            {
                "contamination_or_nu": c,
                "iforest_p_at_k": r["isolation_forest"]["precision_at_k"],
                "ocsvm_p_at_k": r["one_class_svm"]["precision_at_k"],
                "random_p_at_k": r["random_precision_at_k"],
            }
        )
    return rows


def main() -> None:
    primary = evaluate(contamination=0.1, nu=0.1, k=100)
    sens = contamination_sensitivity(k=100)
    out = {"primary": primary, "contamination_sensitivity": sens}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
