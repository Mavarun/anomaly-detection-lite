# anomaly-detection-lite

Lite slice: **IsolationForest** and **OneClassSVM** on synthetic contaminated Gaussian data, evaluated with **precision@k**.

**Not a production claim.** This is a reproducible research/demo slice on planted anomalies — not a guarantee on real-world fraud, intrusion, or sensor streams.

## Hypothesis

1. IsolationForest / OneClassSVM flag anomalies on synthetic contaminated Gaussian data.
2. Precision@k beats random on a labeled anomaly subset.
3. Contamination / `nu` defaults matter — report sensitivity honestly.

## Stack

`pandas` · `numpy` · `scikit-learn` · `scipy` · `pytest`

## Install

```bash
pip install -e ".[dev]"
```

## Run

```bash
pytest -q
python scripts/run_anomaly_slice.py
```

## Measured results (seed=42, n=1000, 100 anomalies, k=100)

| Method | Setting | Precision@k | Random baseline |
|--------|---------|-------------|-----------------|
| IsolationForest | contamination=0.10 | **0.99** | 0.10 |
| OneClassSVM | nu=0.10 | **0.57** | 0.10 |

### Contamination / nu sensitivity (same seed, k=100)

| contamination / nu | IForest P@k | OCSVM P@k | Random |
|--------------------|-------------|-----------|--------|
| 0.05 | 0.99 | 0.46 | 0.10 |
| 0.10 | 0.99 | 0.57 | 0.10 |
| 0.15 | 0.99 | 0.65 | 0.10 |
| 0.20 | 0.99 | 0.76 | 0.10 |

IsolationForest stays near-perfect ranking on this easy planted-offset task. OneClassSVM improves as `nu` rises toward the true prevalence / above it — defaults are not free.

## Layout

```
src/anomaly_lite/   detectors, metrics, synthetic data helpers
tests/              synthetic anomaly + precision@k tests
scripts/            run_anomaly_slice.py
```

## Limits / weaknesses

- Synthetic Gaussians with a large mean offset are an **easy** separation; real anomalies are rarer, overlapping, and non-stationary.
- Precision@k ignores calibration and operating-point costs; flagged counts depend on contamination/`nu`.
- OneClassSVM is sensitive to kernel/`gamma` and scale; no hyperparameter search here.
- Labels are known only because we planted them — unsupervised eval on unlabeled production data needs proxies or delayed labels.
- Single seed in the README table; variance across seeds is not fully characterized.

## License

MIT (demo code).
