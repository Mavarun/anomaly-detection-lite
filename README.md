# anomaly-detection-lite

Research slices on anomaly detection with **honest evaluation**:

1. **Slice 1**: IsolationForest and OneClassSVM on synthetic contaminated Gaussian data, precision@k.
2. **Slice 2 (this PR)**: causal **streaming detectors** (EWMA, CUSUM, robust z), **point vs point-adjusted vs event-level** metrics, **threshold calibration on a held-out window**, an **IsolationForest vs LOF** benchmark on synthetic plus one public dataset, and a CUSUM check on the public **Nile** series.

**Not a production claim.** Reproducible, offline, seeded demos on planted anomalies plus two small public datasets bundled with scikit-learn / statsmodels.

## Stack

`numpy` · `pandas` · `scikit-learn` · `scipy` · `statsmodels` · `pytest`

## Install / run

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_anomaly_slice.py      # slice 1
python scripts/run_streaming_slice.py    # slice 2 (~16 s on CPU)
```

CI: `.github/workflows/tests.yml` runs pytest on push / PR (Python 3.11 and 3.12).

---

## Slice 2 hypotheses

1. On a seasonal, autocorrelated stream, a robust trailing-window z-score catches spikes and variance bursts; CUSUM is needed for small persistent level shifts; no single detector wins on every event type.
2. Point-adjusted (PA) F1 inflates results, enough that a random scorer looks respectable; event-level precision/recall exposes it.
3. Thresholds chosen on a held-out calibration window lose some F1 against a test-tuned "oracle" threshold, and that gap should be reported rather than hidden.
4. LOF beats IsolationForest when anomalies are *local* (near a dense cluster), not when they are global; LOF's `n_neighbors` matters.

## Slice 2 method

- **Stream** (`stream_data.py`): `10 + 2 sin(2 pi t / 96) + AR(1)(phi=0.6, sigma=0.5)`, n=6000, 24 injected non-overlapping events cycling through spike (1-3 pts, 5-8 marginal sd), level shift (20-60 pts, 1.5-3 sd) and variance burst (30-80 pts, innovation sd x3-x4); no events in the first 600 points.
- **Windows**: consecutive fit (30%) / calibration (20%) / test (50%). The seasonal profile (per-phase median) and CUSUM reference come from the fit window; thresholds from the calibration window; every number below is on the test window (~13 events per seed).
- **Detectors** (`streaming.py`, all causal; tested by rewriting the future and checking that past scores do not change): EWMA z (alpha 0.05, Huber-clipped updates), two-sided CUSUM (k = 0.5, MAD-standardised), robust z (median/MAD over a trailing 200-point window that excludes the current point), and a **random** control (i.i.d. uniform scores).
- **Threshold protocols** (`thresholds.py`): `cal_quantile` = finite-sample rank so that 0.5% of label-0 calibration points alarm; `cal_best_f1` = maximise event-F1 on the labelled calibration window; `test_oracle` = maximise event-F1 on the test window itself (optimistic reference only; its grid includes the calibrated thresholds so it upper-bounds them).
- **Metrics** (`event_metrics.py`): point P/R/F1; point-adjusted F1 (Xu et al., 2018); event-level recall (event hit if >= 1 alarm in [start, end + 20]) and precision (share of alarm *segments* touching a tolerance-extended event); detection delay; point false-alarm rate (FAR) on normal points; threshold-free point ROC-AUC.
- **Tabular benchmark** (`benchmark.py`): IsolationForest, LOF (k = 20 and k = 150, standardised), OneClassSVM on `global_offset` (slice-1 data), `local_density` (tight + diffuse clusters, anomalies in a shell near the tight cluster), and `breast_cancer` (public sklearn WDBC in the ODDS-style setup: 357 benign inliers + 21 seeded-random malignant rows). Unsupervised transductive fit; labels only used for ROC-AUC, AP and precision@n_anomalies.
- **Nile** (`stream_eval.nile_cusum_alarms`): annual Aswan flow 1871-1970 from `statsmodels.datasets.nile`; textbook level drop around 1898-1899 (Cobb, 1978).
- 10 seeds throughout; tables report mean ± std across seeds.

## Slice 2 results

Test-window events per seed: 13.2 (min 12, max 15)

### Streaming detectors on the test window (mean ± std, 10 seeds; tolerance 20, target FAR 0.5%)

| detector | protocol | event F1 | event P | event R | PA-F1 | point F1 | point FAR | delay (steps) | point ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ewma | cal_quantile | 0.769 ± 0.065 | 0.699 ± 0.115 | 0.874 ± 0.072 | 0.894 ± 0.063 | 0.122 ± 0.022 | 0.004 ± 0.002 | 2.6 ± 1.9 | 0.610 ± 0.020 |
| ewma | cal_best_f1 | 0.690 ± 0.166 | 0.805 ± 0.167 | 0.689 ± 0.243 | 0.720 ± 0.280 | 0.087 ± 0.052 | 0.003 ± 0.003 | 1.9 ± 1.6 | 0.610 ± 0.020 |
| ewma | test_oracle | 0.842 ± 0.060 | 0.841 ± 0.097 | 0.859 ± 0.100 | 0.883 ± 0.101 | 0.104 ± 0.029 | 0.002 ± 0.001 | 3.1 ± 2.0 | 0.610 ± 0.020 |
| cusum | cal_quantile | 0.315 ± 0.260 | 0.630 ± 0.447 | 0.230 ± 0.209 | 0.268 ± 0.261 | 0.103 ± 0.117 | 0.064 ± 0.091 | 32.9 ± 10.2 | 0.714 ± 0.050 |
| cusum | cal_best_f1 | 0.748 ± 0.109 | 0.758 ± 0.130 | 0.751 ± 0.130 | 0.540 ± 0.054 | 0.342 ± 0.078 | 0.285 ± 0.098 | 12.8 ± 5.7 | 0.714 ± 0.050 |
| cusum | test_oracle | 0.860 ± 0.054 | 0.862 ± 0.107 | 0.876 ± 0.094 | 0.519 ± 0.060 | 0.371 ± 0.047 | 0.333 ± 0.071 | 10.2 ± 4.6 | 0.714 ± 0.050 |
| robust_z | cal_quantile | 0.903 ± 0.030 | 0.887 ± 0.053 | 0.925 ± 0.061 | 0.934 ± 0.034 | 0.402 ± 0.069 | 0.005 ± 0.003 | 3.0 ± 1.6 | 0.821 ± 0.015 |
| robust_z | cal_best_f1 | 0.865 ± 0.058 | 0.940 ± 0.081 | 0.820 ± 0.132 | 0.870 ± 0.094 | 0.315 ± 0.104 | 0.002 ± 0.004 | 3.3 ± 1.5 | 0.821 ± 0.015 |
| robust_z | test_oracle | 0.937 ± 0.028 | 0.917 ± 0.061 | 0.963 ± 0.050 | 0.960 ± 0.035 | 0.365 ± 0.109 | 0.004 ± 0.006 | 4.1 ± 1.9 | 0.821 ± 0.015 |
| random | cal_quantile | 0.198 ± 0.064 | 0.189 ± 0.055 | 0.222 ± 0.085 | 0.270 ± 0.179 | 0.007 ± 0.005 | 0.006 ± 0.002 | 33.8 ± 13.9 | 0.502 ± 0.010 |
| random | cal_best_f1 | 0.352 ± 0.063 | 0.236 ± 0.044 | 0.774 ± 0.264 | 0.632 ± 0.165 | 0.090 ± 0.072 | 0.112 ± 0.149 | 14.5 ± 11.6 | 0.502 ± 0.010 |
| random | test_oracle | 0.417 ± 0.029 | 0.276 ± 0.042 | 0.919 ± 0.153 | 0.681 ± 0.138 | 0.144 ± 0.075 | 0.177 ± 0.168 | 8.3 ± 7.9 | 0.502 ± 0.010 |

### Event recall by injected type (held-out thresholds)

| detector | protocol | spike | level shift | variance burst |
| --- | --- | ---: | ---: | ---: |
| ewma | cal_quantile | 1.00 | 0.63 | 1.00 |
| ewma | cal_best_f1 | 0.95 | 0.39 | 0.78 |
| cusum | cal_quantile | 0.03 | 0.38 | 0.26 |
| cusum | cal_best_f1 | 0.27 | 0.95 | 0.92 |
| robust_z | cal_quantile | 1.00 | 0.78 | 1.00 |
| robust_z | cal_best_f1 | 1.00 | 0.51 | 0.98 |
| random | cal_quantile | 0.12 | 0.13 | 0.34 |
| random | cal_best_f1 | 0.64 | 0.75 | 0.87 |

### Tabular benchmark: ranking quality (mean ± std, 10 seeds)

| dataset | detector | ROC-AUC | average precision | P@n_anom |
| --- | --- | ---: | ---: | ---: |
| global_offset | isolation_forest | 1.000 ± 0.000 | 0.999 ± 0.002 | 0.997 ± 0.009 |
| global_offset | lof_k20 | 0.406 ± 0.060 | 0.084 ± 0.014 | 0.061 ± 0.038 |
| global_offset | lof_k150 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| global_offset | ocsvm | 0.881 ± 0.022 | 0.598 ± 0.028 | 0.536 ± 0.015 |
| local_density | isolation_forest | 0.957 ± 0.008 | 0.390 ± 0.053 | 0.375 ± 0.063 |
| local_density | lof_k20 | 0.998 ± 0.002 | 0.952 ± 0.032 | 0.885 ± 0.065 |
| local_density | lof_k150 | 0.999 ± 0.001 | 0.970 ± 0.045 | 0.980 ± 0.016 |
| local_density | ocsvm | 0.948 ± 0.007 | 0.318 ± 0.026 | 0.320 ± 0.016 |
| breast_cancer | isolation_forest | 0.936 ± 0.017 | 0.575 ± 0.047 | 0.548 ± 0.046 |
| breast_cancer | lof_k20 | 0.912 ± 0.014 | 0.414 ± 0.066 | 0.433 ± 0.091 |
| breast_cancer | lof_k150 | 0.939 ± 0.019 | 0.573 ± 0.052 | 0.552 ± 0.060 |
| breast_cancer | ocsvm | 0.848 ± 0.027 | 0.242 ± 0.058 | 0.243 ± 0.069 |

Seeds where LOF beats IsolationForest on AP: breast_cancer: k20 0.0, k150 0.5; global_offset: k20 0.0, k150 0.1; local_density: k20 1.0, k150 1.0

### Nile (statsmodels, 1871-1970): first two-sided CUSUM alarm (k = 0.5)

| reference years | reference ends | h | ref median | ref robust sd | first alarm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | 1885 | 4 | 1120 | 148.3 | 1901 |
| 15 | 1885 | 5 | 1120 | 148.3 | 1902 |
| 20 | 1890 | 4 | 1115 | 155.7 | 1901 |
| 20 | 1890 | 5 | 1115 | 155.7 | 1902 |
| 25 | 1895 | 4 | 1140 | 133.4 | 1888 |
| 25 | 1895 | 5 | 1140 | 133.4 | 1889 |

### Reading the results

- **H1 mostly supported.** Robust z with a held-out 0.5% FAR threshold gave the best held-out event F1 (0.903 ± 0.030, realised point FAR 0.5%), with spike and variance-burst recall 1.00 and level-shift recall 0.78. EWMA was close on spikes and bursts but weaker on level shifts (0.63). CUSUM ranked level shifts and bursts best under a supervised threshold (recall 0.95 / 0.92) but missed most spikes (0.27).
- **CUSUM's event precision looks fine, but its point FAR does not.** With `cal_best_f1` CUSUM keeps alarming after events end, so 28.5% of normal test points are alarmed (point FAR 0.285). Event precision counts segments, and those long post-event runs touch a (tolerance-extended) event, so event precision alone hides this. Its FAR-quantile threshold failed outright (event F1 0.315 ± 0.260): accumulated scores on the calibration window are not exchangeable with test scores.
- **H2 supported.** The random control reaches PA-F1 0.63-0.68 under F1-tuned thresholds, against event F1 0.35-0.42 and point F1 0.09-0.14. PA-F1 is reported here only beside stricter metrics.
- **H3 supported, with a twist.** Test-oracle thresholds add +0.03 to +0.11 event F1 over the best held-out protocol. Supervised `cal_best_f1` was *worse* than the unsupervised FAR quantile for EWMA and robust z: the calibration window holds only ~5 events, so F1-tuning on it overfits.
- **H4 supported, with a caveat.** LOF k=20 beat IsolationForest on `local_density` in 10/10 seeds (AP 0.952 vs 0.390) but collapsed on `global_offset` (ROC-AUC 0.406). There the 100 planted anomalies form their own cluster, which k=20 treats as normal (masking). k=150 fixes it, but that k was chosen knowing the anomaly count. On public breast-cancer data, IsolationForest beat LOF k=20 in 10/10 seeds (AP 0.575 vs 0.414); LOF k=150 tied IF (0.573, 5/10 seeds).
- **Nile:** with a 15- or 20-year reference, CUSUM first alarms in 1901 (h = 4) or 1902 (h = 5), 3-4 years after the textbook 1898 break. With a 25-year reference ending in 1895, the tighter scale triggers an alarm in 1888, *inside* the reference window (a Phase-I in-sample artefact, not an online detection). The result depends on the choice of reference.

### Test design (slice 2)

- Detectors: causality (past scores unchanged when the future is rewritten), spike score above the 99th background percentile, CUSUM accumulates a 1-sd shift that robust z misses (and keeps alarming after it ends), reset scheme, EWMA clipping prevents a huge spike masking a moderate one 20 steps later, zero-MAD handling, seasonal profile removal.
- Metrics: hand-built cases for point vs PA vs event scoring, tolerance counts late but not early alarms, false-alarm rate arithmetic, and a random-alarm test reproducing the PA-inflation critique (PA-F1 > 0.5 with point F1 < 0.1).
- Thresholds: hand-computed quantile rank; realised FAR on i.i.d. scores averages to target (200 repeats); calibrated thresholds do not change when test labels are rewritten; oracle upper-bounds both calibrated protocols.
- Benchmark: LOF transductive/novelty modes, LOF > IF on local anomalies, LOF k=20 masking vs k=150, breast-cancer loader reproducibility.
- End to end: structure, random control ROC-AUC ~0.5, robust z beats random by > 0.3 event F1, PA flatters random, Nile alarm within 1898-1905 for the 20-year reference and earlier-than-1898 for the 25-year one.

## Limits / weaknesses

- The stream is synthetic, with a fixed seasonal period, a stationary AR(1) and cleanly separated events. Real streams have drift, missing data, multiple seasonalities and overlapping incidents.
- The calibration window holds ~5 events, so supervised threshold tuning is noisy (cal_best_f1 std up to 0.24 on recall). A longer or rolling calibration would be needed.
- Event-level precision counts alarm segments, not alarmed time. Read it together with point FAR (CUSUM is the cautionary example). Tolerance (20 steps) is a free parameter that favours lagging detectors.
- CUSUM has no reset in the evaluated configuration and is standardised by a fixed fit-window MAD. EWMA slowly absorbs persistent level shifts. Robust z treats a long event as the new baseline once it fills its window. No detector models variance explicitly.
- Tabular benchmark: one public dataset (n = 378, 21 anomalies), so seed variance includes which malignant rows are drawn. `contamination`/`nu` are set to the true prevalence (this only affects binary predictions, not the ranking metrics reported). LOF k=150 is an informed choice, not a tuned one.
- Nile is a single 100-point series with one well-known break. It illustrates reference-window sensitivity and proves nothing about detector quality.
- Slice-1 limits still apply (easy planted offsets, single-seed table below).

---

## Slice 1 (prior): IsolationForest / OneClassSVM precision@k

### Hypothesis

1. IsolationForest / OneClassSVM flag anomalies on synthetic contaminated Gaussian data.
2. Precision@k beats random on a labeled anomaly subset.
3. Contamination / `nu` defaults matter -- report sensitivity honestly.

### Measured results (seed=42, n=1000, 100 anomalies, k=100)

| Method | Setting | Precision@k | Random baseline |
|--------|---------|-------------|-----------------|
| IsolationForest | contamination=0.10 | **0.99** | 0.10 |
| OneClassSVM | nu=0.10 | **0.57** | 0.10 |

#### Contamination / nu sensitivity (same seed, k=100)

| contamination / nu | IForest P@k | OCSVM P@k | Random |
|--------------------|-------------|-----------|--------|
| 0.05 | 0.99 | 0.46 | 0.10 |
| 0.10 | 0.99 | 0.57 | 0.10 |
| 0.15 | 0.99 | 0.65 | 0.10 |
| 0.20 | 0.99 | 0.76 | 0.10 |

IsolationForest ranks almost perfectly on this easy planted-offset task. OneClassSVM improves as `nu` rises toward or above the true prevalence, so defaults are not free.

## Layout

```
src/anomaly_lite/
  data.py            # slice 1 contaminated Gaussian
  detectors.py       # IsolationForest, OneClassSVM, LOF wrappers
  metrics.py         # precision@k, random baseline
  stream_data.py     # labelled seasonal AR(1) stream + contiguous windows
  streaming.py       # causal EWMA / CUSUM / robust-z + seasonal profile
  event_metrics.py   # point, point-adjusted, event-level metrics
  thresholds.py      # held-out FAR quantile, best-F1, oracle protocols
  benchmark.py       # IF vs LOF vs OCSVM tabular benchmark (incl. breast cancer)
  stream_eval.py     # end-to-end streaming harness + Nile CUSUM
scripts/             # run_anomaly_slice.py, run_streaming_slice.py
tests/
```

## License

MIT (demo code).
