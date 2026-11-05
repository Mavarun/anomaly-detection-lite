# anomaly-detection-lite

Research slices on anomaly detection with **honest evaluation**:

1. **Slice 1**: IsolationForest and OneClassSVM on synthetic contaminated Gaussian data, precision@k.
2. **Slice 2**: causal **streaming detectors** (EWMA, CUSUM, robust z), **point vs point-adjusted vs event-level** metrics, **threshold calibration on a held-out window**, an **IsolationForest vs LOF** benchmark on synthetic plus one public dataset, and a CUSUM check on the public **Nile** series.
3. **Slice 3**: the same stream under **slow drift and 5% missing data**; a causal **online de-seasonaliser**, an explicit **variance detector**, gap-skipping detectors, **walk-forward thresholds** (label-free rolling quantile and a rescaled held-out threshold) and an **alarm-cost model**, all scored out of sample with paired per-seed effects.
4. **Slice 4 (this PR)**: a **second (weekly) seasonality**, a causal **harmonic RLS de-seasonaliser**, a **freezable-reference CUSUM** and a **contamination-robust rolling threshold** (bulk-tail extrapolation), tested against the slice-3 baselines, with the CUSUM allowance re-examined.

**Not a production claim.** Reproducible, offline, seeded demos on planted anomalies plus two small public datasets bundled with scikit-learn / statsmodels.

## Stack

`numpy` · `pandas` · `scikit-learn` · `scipy` · `statsmodels` · `pytest`

## Install / run

```bash
pip install -e ".[dev]"
pytest -q
python scripts/run_anomaly_slice.py      # slice 1
python scripts/run_streaming_slice.py    # slice 2 (~16 s on CPU)
python scripts/run_drift_slice.py        # slice 3 (~30 s on CPU)
python scripts/run_slice4.py             # slice 4 (~2 min on CPU)
```

CI: `.github/workflows/tests.yml` runs pytest on push / PR (Python 3.11 and 3.12).

---

## Slice 4 hypotheses

1. A second, longer cycle (weekly over daily) leaks into a one-period de-seasonaliser as level drift. Two sin/cos terms per extra period, fitted by robust recursive least squares, should remove it at no cost when there is no such cycle.
2. The plain trailing-rank rolling threshold of slice 3 is too conservative because ~5% anomalies sit in its own window. A tail model fitted to the *bulk* quantiles (80% / 90%) should restore the false-alarm rate without labels.
3. Slice 3 called CUSUM unusable under drift. Part of that may be the reference (a fixed one drifts away; an adaptive one absorbs the shift). A reference that adapts only while calm and freezes while alarmed should fix both.
4. A second suspect for CUSUM's weakness: with AR(1) noise (phi = 0.6) the textbook allowance k = 0.5 is far too small.

## Slice 4 method

- **Stream**: slice-3 `drift` (trend, random-walk baseline, +50% seasonal amplitude) plus, for `weekly` and `weekly_missing`, a 672-step sinusoid of amplitude 1.5 (`season2_period`, `season2_amp`; off by default, so slices 1-3 streams are unchanged). `weekly_missing` also drops 5% of points in gaps. Same events and labels as before; the cycle is not an anomaly.
- **De-seasonalisers** (all initialised on the fit window, then causal): `static` and `online` as in slice 3; `harm_d` = `OnlineHarmonicProfile` with period 96; `harm_dw` = periods 96 and 672. The profile is a level plus one sin/cos pair per period, Huber-IRLS on the fit window, then RLS with forgetting 0.995 (memory ~200 steps) on innovations clipped at 2 sigma.
- **Detectors**: EWMA, robust z, the slice-2 fixed-reference CUSUM at k = 0.5 (`cusum`) and k = 1.5 (`cusum_k15`), `FrozenReferenceCUSUM` at the same two allowances (reference mean/scale adapt only while both sums are below 2, re-anchor after 150 frozen steps) and the slice-3 variance detector.
- **Thresholds** (target FAR 0.5%): `static_quantile`, `rolling_quantile` (plain trailing rank) and `rolling_rescaled` as in slice 3; `rolling_bulk` = label-free trailing window of 1,200, refreshed every 50 steps, threshold from a half-normal tail (EWMA, robust z) or exponential tail (CUSUM, variance) fitted to the 80% and 90% quantiles. The tail family follows the score type and was fixed before the evaluation seeds were run.
- **Seeds**: hyper-parameters (forgetting 0.995, anchors 80/90, freeze level 2, k = 1.5) were chosen on development seeds 100-105 only. All tables are 10 evaluation seeds (0-9), out of sample on the test window. Costs, FAR and paired statistics are defined as in slice 3.

## Slice 4 results

### Event F1 / point FAR / cost per 1k by de-seasonaliser, held-out FAR-quantile threshold (test window, mean ± std, 10 seeds)

| scenario | detector | static F1 | online F1 | harm_d F1 | harm_dw F1 | online FAR | harm_dw FAR | cost/1k online | cost/1k harm_dw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| drift | ewma | 0.721 ± 0.052 | 0.789 ± 0.057 | 0.785 ± 0.063 | 0.797 ± 0.055 | 0.004 | 0.004 | 9.2 | 8.8 |
| drift | robust_z | 0.869 ± 0.036 | 0.908 ± 0.024 | 0.895 ± 0.032 | 0.897 ± 0.022 | 0.004 | 0.005 | 7.4 | 7.9 |
| drift | cusum | 0.780 ± 0.329 | 0.139 ± 0.206 | 0.106 ± 0.165 | 0.157 ± 0.212 | 0.015 | 0.008 | 41.4 | 41.1 |
| drift | cusum_k15 | 0.742 ± 0.238 | 0.422 ± 0.311 | 0.420 ± 0.295 | 0.502 ± 0.216 | 0.009 | 0.004 | 33.3 | 30.6 |
| drift | cusum_frozen | 0.212 ± 0.204 | 0.214 ± 0.194 | 0.237 ± 0.245 | 0.329 ± 0.242 | 0.021 | 0.046 | 39.9 | 37.3 |
| drift | cusum_frozen_k15 | 0.348 ± 0.280 | 0.529 ± 0.267 | 0.542 ± 0.221 | 0.596 ± 0.205 | 0.006 | 0.008 | 29.2 | 25.9 |
| drift | variance | 0.545 ± 0.107 | 0.540 ± 0.107 | 0.536 ± 0.101 | 0.533 ± 0.099 | 0.011 | 0.010 | 29.7 | 30.0 |
| weekly | ewma | 0.729 ± 0.078 | 0.800 ± 0.049 | 0.779 ± 0.040 | 0.809 ± 0.039 | 0.003 | 0.004 | 10.3 | 8.4 |
| weekly | robust_z | 0.782 ± 0.084 | 0.798 ± 0.080 | 0.827 ± 0.064 | 0.897 ± 0.023 | 0.007 | 0.006 | 16.2 | 7.9 |
| weekly | cusum | 0.679 ± 0.305 | 0.262 ± 0.249 | 0.264 ± 0.256 | 0.108 ± 0.197 | 0.113 | 0.000 | 38.3 | 41.9 |
| weekly | cusum_k15 | 0.454 ± 0.207 | 0.398 ± 0.290 | 0.509 ± 0.241 | 0.790 ± 0.147 | 0.011 | 0.003 | 31.8 | 15.8 |
| weekly | cusum_frozen | 0.097 ± 0.098 | 0.063 ± 0.108 | 0.075 ± 0.118 | 0.194 ± 0.185 | 0.021 | 0.028 | 43.8 | 40.5 |
| weekly | cusum_frozen_k15 | 0.203 ± 0.176 | 0.197 ± 0.282 | 0.170 ± 0.228 | 0.599 ± 0.208 | 0.025 | 0.008 | 39.3 | 25.5 |
| weekly | variance | 0.522 ± 0.094 | 0.531 ± 0.098 | 0.537 ± 0.101 | 0.533 ± 0.099 | 0.011 | 0.011 | 29.9 | 29.9 |
| weekly_missing | ewma | 0.735 ± 0.060 | 0.769 ± 0.058 | 0.791 ± 0.051 | 0.812 ± 0.061 | 0.005 | 0.005 | 10.7 | 7.9 |
| weekly_missing | robust_z | 0.770 ± 0.109 | 0.785 ± 0.091 | 0.815 ± 0.062 | 0.891 ± 0.028 | 0.008 | 0.005 | 16.8 | 8.1 |
| weekly_missing | cusum | 0.591 ± 0.242 | 0.268 ± 0.246 | 0.270 ± 0.239 | 0.132 ± 0.262 | 0.117 | 0.002 | 37.9 | 40.8 |
| weekly_missing | cusum_k15 | 0.426 ± 0.197 | 0.403 ± 0.300 | 0.449 ± 0.281 | 0.773 ± 0.145 | 0.014 | 0.003 | 31.1 | 16.5 |
| weekly_missing | cusum_frozen | 0.087 ± 0.083 | 0.086 ± 0.120 | 0.158 ± 0.182 | 0.064 ± 0.139 | 0.035 | 0.018 | 42.7 | 42.8 |
| weekly_missing | cusum_frozen_k15 | 0.178 ± 0.183 | 0.230 ± 0.277 | 0.247 ± 0.244 | 0.531 ± 0.173 | 0.034 | 0.006 | 38.5 | 29.2 |
| weekly_missing | variance | 0.507 ± 0.115 | 0.519 ± 0.108 | 0.513 ± 0.108 | 0.512 ± 0.109 | 0.011 | 0.011 | 30.4 | 30.6 |

### Threshold protocols, weekly scenario, harm_dw residuals (mean over 10 seeds)

| detector | protocol | event F1 | point FAR | cost / 1k |
| --- | --- | ---: | ---: | ---: |
| ewma | static_quantile | 0.809 | 0.0040 | 8.4 |
| ewma | rolling_quantile | 0.712 | 0.0003 | 19.2 |
| ewma | rolling_rescaled | 0.807 | 0.0049 | 8.0 |
| ewma | rolling_bulk | 0.812 | 0.0040 | 8.1 |
| robust_z | static_quantile | 0.897 | 0.0055 | 7.9 |
| robust_z | rolling_quantile | 0.646 | 0.0000 | 24.1 |
| robust_z | rolling_rescaled | 0.886 | 0.0064 | 8.4 |
| robust_z | rolling_bulk | 0.893 | 0.0020 | 8.7 |
| cusum | static_quantile | 0.108 | 0.0001 | 41.9 |
| cusum | rolling_quantile | 0.330 | 0.0024 | 37.5 |
| cusum | rolling_rescaled | 0.190 | 0.1281 | 54.7 |
| cusum | rolling_bulk | 0.334 | 0.0052 | 36.3 |
| cusum_k15 | static_quantile | 0.790 | 0.0028 | 15.8 |
| cusum_k15 | rolling_quantile | 0.487 | 0.0009 | 31.6 |
| cusum_k15 | rolling_rescaled | 0.790 | 0.0028 | 15.8 |
| cusum_k15 | rolling_bulk | 0.929 | 0.0050 | 6.4 |
| cusum_frozen | static_quantile | 0.194 | 0.0277 | 40.5 |
| cusum_frozen | rolling_quantile | 0.299 | 0.0363 | 38.8 |
| cusum_frozen | rolling_rescaled | 0.170 | 0.0286 | 41.4 |
| cusum_frozen | rolling_bulk | 0.185 | 0.0155 | 41.1 |
| cusum_frozen_k15 | static_quantile | 0.599 | 0.0080 | 25.5 |
| cusum_frozen_k15 | rolling_quantile | 0.290 | 0.0013 | 37.3 |
| cusum_frozen_k15 | rolling_rescaled | 0.599 | 0.0080 | 25.5 |
| cusum_frozen_k15 | rolling_bulk | 0.577 | 0.0060 | 26.9 |
| variance | static_quantile | 0.533 | 0.0109 | 29.9 |
| variance | rolling_quantile | 0.383 | 0.0006 | 35.8 |
| variance | rolling_rescaled | 0.508 | 0.0365 | 31.2 |
| variance | rolling_bulk | 0.351 | 0.0021 | 35.6 |

### Paired per-seed effects (10 seeds; bootstrap 95% CI; exact sign test)

| comparison (b - a) | scenario | detector | metric | mean diff | 95% CI | seeds with diff > 0 | sign p |
| --- | --- | --- | --- | ---: | :---: | ---: | ---: |
| harm_dw - online | drift | ewma | event_f1 | +0.008 | [-0.028, +0.043] | 6/10 | 0.754 |
| harm_dw - online | drift | robust_z | event_f1 | -0.010 | [-0.033, +0.011] | 5/10 | 1.000 |
| harm_dw - online | weekly | ewma | event_f1 | +0.009 | [-0.029, +0.048] | 5/10 | 1.000 |
| harm_dw - online | weekly | robust_z | event_f1 | +0.099 | [+0.051, +0.147] | 9/10 | 0.021 |
| harm_dw - online | drift | ewma | cost_per_1k | -0.363 | [-1.827, +1.089] | 5/10 | 1.000 |
| harm_dw - online | drift | robust_z | cost_per_1k | +0.455 | [-0.845, +1.846] | 4/10 | 0.754 |
| harm_dw - online | weekly | ewma | cost_per_1k | -1.937 | [-4.332, +0.356] | 3/10 | 0.344 |
| harm_dw - online | weekly | robust_z | cost_per_1k | -8.270 | [-11.692, -4.765] | 1/10 | 0.021 |
| harm_dw - harm_d | drift | robust_z | event_f1 | +0.002 | [-0.011, +0.018] | 4/10 | 0.754 |
| harm_dw - harm_d | weekly | robust_z | event_f1 | +0.071 | [+0.028, +0.113] | 9/10 | 0.021 |
| rolling_bulk - static thr | weekly | ewma | event_f1 | +0.003 | [-0.016, +0.023] | 5/10 | 1.000 |
| rolling_bulk - static thr | weekly | robust_z | event_f1 | -0.004 | [-0.027, +0.018] | 5/10 | 1.000 |
| rolling_bulk - static thr | weekly | cusum_k15 | event_f1 | +0.139 | [+0.063, +0.220] | 7/10 | 0.016 |
| rolling_bulk - static thr | weekly | variance | event_f1 | -0.183 | [-0.320, -0.069] | 1/10 | 0.125 |
| rolling_bulk - plain rolling rank | weekly | ewma | event_f1 | +0.100 | [+0.018, +0.187] | 7/10 | 0.344 |
| rolling_bulk - plain rolling rank | weekly | robust_z | event_f1 | +0.247 | [+0.201, +0.296] | 10/10 | 0.002 |
| rolling_bulk - rescaled | weekly | ewma | event_f1 | +0.005 | [-0.044, +0.050] | 5/10 | 1.000 |
| rolling_bulk - rescaled | weekly | robust_z | event_f1 | +0.007 | [-0.019, +0.030] | 6/10 | 0.754 |
| CUSUM k 1.5 - k 0.5 | drift | see comparison | event_f1 | +0.346 | [+0.185, +0.507] | 8/10 | 0.039 |
| CUSUM k 1.5 - k 0.5 | weekly | see comparison | event_f1 | +0.682 | [+0.502, +0.836] | 10/10 | 0.002 |
| frozen - fixed ref (k 1.5) | drift | see comparison | event_f1 | +0.093 | [-0.076, +0.301] | 5/10 | 1.000 |
| frozen - fixed ref (k 1.5) | weekly | see comparison | event_f1 | -0.192 | [-0.248, -0.133] | 0/10 | 0.002 |
| frozen - fixed ref (k 0.5) | drift | see comparison | event_f1 | +0.173 | [+0.046, +0.328] | 6/10 | 0.289 |
| frozen - fixed ref (k 0.5) | weekly | see comparison | event_f1 | +0.086 | [+0.011, +0.178] | 4/10 | 0.688 |
| CUSUM k 1.5 - robust z | drift | see comparison | event_f1 | -0.395 | [-0.542, -0.279] | 0/10 | 0.002 |
| CUSUM k 1.5 - robust z | weekly | see comparison | event_f1 | -0.107 | [-0.191, -0.030] | 3/10 | 0.344 |

### Forgetting factor sensitivity (harm_dw, weekly, static threshold, robust z / cusum_k15 event F1)

| forgetting | memory (steps) | robust_z F1 | robust_z cost/1k | cusum_k15 F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.98 | 50 | 0.884 | 9.2 | 0.822 |
| 0.99 | 100 | 0.900 | 7.9 | 0.794 |
| 0.995 | 200 | 0.897 | 7.9 | 0.790 |
| 0.999 | 1000 | 0.891 | 8.2 | 0.770 |

### Reading the results

- **H1 supported where the cycle exists, neutral where it does not.** With a weekly cycle, robust z with the one-period online profile drops from 0.908 (drift only) to 0.798 event F1 (cost per 1k 7.4 to 16.2). Adding the 672-step terms brings it to 0.897 (+0.099, CI [+0.051, +0.147], 9/10 seeds, p = 0.021; cost 7.9, -8.3 per 1k). Without a weekly cycle the extra terms change nothing measurable (robust z -0.010, CI [-0.033, +0.011]; EWMA +0.008). EWMA gains little (+0.009, CI covers 0), because it adapts quickly to a slow cycle by itself. A daily-only harmonic profile (`harm_d`) is only modestly better than the per-phase `online` profile (0.827 vs 0.798 on weekly) and clearly behind `harm_dw` (-0.071 vs it, 9/10 seeds); the benefit is mainly the second period, not the RLS.
- **H2 supported for z-type scores.** The plain rolling rank gives robust z a point FAR of 0.0000 and F1 0.646 on `weekly`. The bulk-tail rolling threshold gives 0.893 at point FAR 0.0020 with no labels (+0.247 over the plain rank, 10/10 seeds, p = 0.002) and ties both the static held-out threshold (-0.004, CI [-0.027, +0.018]) and the rescaled one (+0.007). It also helps the accumulated CUSUM score (`cusum_k15` 0.929 vs 0.790, +0.139, 7/10 seeds, p = 0.016). **It hurts the variance detector** (0.351 vs 0.533, -0.183, 1/10 seeds) because that log-ratio score is neither half-normal nor exponential in the tail; it is a model, and the realised FAR (0.002) shows it is conservative there. For EWMA the gain over the plain rank is +0.100 (CI [+0.018, +0.187], 7/10 seeds, p = 0.344).
- **H3 rejected.** Freezing the reference does not rescue CUSUM. At k = 0.5 the frozen version is still unusable (0.194 on `weekly`, 0.329 on `drift`, both with huge variance; vs 0.108 and 0.157 fixed). At k = 1.5 it is worse than the fixed reference on `weekly` (-0.192, CI [-0.248, -0.133], 0/10 seeds, p = 0.002) and indistinguishable on `drift` (+0.093, CI [-0.076, +0.301]). Once the residuals are de-seasonalised properly there is little drift left for an adaptive reference to remove, and freezing delays the re-anchor after real shifts. `FrozenReferenceCUSUM` stays in the package as a tested but not recommended option.
- **H4 supported, and it is the main CUSUM finding.** Raising the allowance from 0.5 to 1.5 lifts CUSUM event F1 by +0.346 on `drift` (8/10 seeds, p = 0.039) and +0.682 on `weekly` (10/10, p = 0.002) at `harm_dw`: 0.108 to 0.790. With the bulk-tail rolling threshold `cusum_k15` reaches 0.929 and cost 6.4 per 1k, the best number in the table on `weekly`. Slice 3's "CUSUM is unusable under drift" was therefore mostly a tuning statement about k = 0.5 on autocorrelated noise plus a drifting fit-once reference. Caveats: on `drift` (no weekly cycle) `cusum_k15` is still weak and noisy (0.502 ± 0.216 at `harm_dw`) while robust z stays at 0.897, so CUSUM at k = 1.5 is still behind robust z there (-0.395, 0/10 seeds); on `weekly` the gap is -0.107 (CI [-0.191, -0.030], 3/10, p = 0.344). The k = 1.5 value was chosen on development seeds and the noise model is AR(1) by construction.
- **Forgetting factor**: 0.98 to 0.999 moves robust z F1 only between 0.884 and 0.900; CUSUM prefers shorter memory (0.822 at 0.98 vs 0.770 at 0.999), i.e. faster tracking also absorbs the per-step noise structure. 0.995 is a compromise, not an optimum.

### Test design (slice 4)

- Stream: `season2_*` off reproduces the old stream bit-for-bit; on, it is an exact additive sinusoid and leaves labels untouched.
- `OnlineHarmonicProfile`: coefficient recovery on a noiseless-ish series, weekly cycle removal where the one-period profile leaves sd > 0.55, growing amplitude plus ramp tracking, causality (future rewrite), a 30-sd spike barely moves the state, a 40-step 10-sd shift is not absorbed within the event, gaps and validation.
- `FrozenReferenceCUSUM`: a 4-sd ramp that sends the fixed-reference CUSUM to alarm on >50% of points, freezing keeps a moderate shift visible, re-anchoring after a permanent shift, causality, gaps.
- `bulk_tail_threshold`: closed forms on exponential and |Gaussian| data, 5% contamination, scale drift, missing placeholders, causality, small samples.
- Harness: row structure, reproducibility, test labels cannot move any alarm for any protocol, no look-ahead for all four de-seasonalisers, a 3-seed check of the harmonic benefit, and a 4-seed re-check of every directional claim above.

### Limits (slice 4)

- The weekly cycle is an exact sinusoid at a known period. Real streams have non-sinusoidal weekly shapes (weekend dips), holidays and drifting periods; the harmonic profile would need more harmonics and a period estimate.
- The bulk-tail threshold relies on anomalies filling less than 10% of the window and on the right tail family; the variance detector shows it can fail. Realised FAR must be monitored.
- CUSUM at k = 1.5 was tuned on the AR(1) coefficient of this simulator; its F1 has a std of 0.15-0.25 across seeds and is noticeably worse on the no-weekly stream. The freezable reference did not help and is untuned beyond a 3 x 3 grid on development seeds.
- Still synthetic, one noise model, no overlapping incidents, no multi-stream correlation.

---

## Slice 3 hypotheses

1. Slow drift (a level trend, a wandering baseline, a growing seasonal amplitude) breaks detectors that depend on a fit-once reference. A causal online de-seasonaliser recovers most of the loss and costs little on a stationary stream.
2. A threshold fixed on one held-out window drifts out of calibration. A walk-forward, label-free recalibration should keep the false-alarm rate near target.
3. An explicit variance detector adds variance-burst detection that mean detectors miss.
4. 5% missing data in short gaps costs little if detectors skip gaps instead of imputing them.

## Slice 3 method

- **Stream** (`make_drifting_stream`): the slice-2 stream (same events, same seed) plus a linear level trend of 3 marginal sd over the series, a Gaussian random-walk baseline (step 0.02 sd), and a seasonal amplitude that grows 50%. `drift_missing` also removes 5% of points in gaps of 5-30 (never an event's first point). Labels are unchanged: drift is not an anomaly.
- **Windows and evaluation**: fit 30% / cal 20% / test 50%, in time order. Every number below is out of sample on the test window, 10 seeds, mean ± std.
- **De-seasonalising**: `static` is slice 2's per-phase median profile, fitted once. `online` (`OnlineSeasonalProfile`) is a causal additive Holt-Winters update (level alpha 0.005, season gamma 0.1) with Huber-clipped innovations (2 sigma), initialised on fit and run forward from cal.
- **Detectors** (all `missing="skip"`: a gap scores 0 and leaves the state untouched): EWMA, CUSUM, robust z, the new `RollingVarianceDetector` (log ratio of short- to long-window MAD-scale of first differences), `rz_or_var` (union of robust-z and variance alarms, each at half the rate) and a random control.
- **Thresholds** (target FAR 0.5%): `static_quantile` is slice 2's `cal_quantile`, fixed for the whole test window. `rolling_quantile` is the same rank on the trailing 1,200 observed scores, refreshed every 50 steps, with no labels. `rolling_rescaled` is the static threshold times the trailing-median / cal-median score.
- **Cost** (`alarm_cost`): 1 per false page, 10 per missed event, 0.1 per step of detection delay and 0.02 per alarmed normal step. Without the last term, an alarm that never clears costs almost nothing. These weights are assumptions; components are kept in the long frame.
- **Paired effects**: per-seed differences, percentile bootstrap 95% CI over seeds, exact sign test.

## Slice 3 results

### Held-out FAR-quantile threshold (static_quantile): fit-once vs online de-seasonalising (test window, mean ± std, 10 seeds)

| scenario | detector | event F1, fit-once | event F1, online | point FAR, fit-once | point FAR, online | cost / 1k, fit-once | cost / 1k, online |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stationary | ewma | 0.769 ± 0.065 | 0.790 ± 0.061 | 0.004 | 0.003 | 9.9 ± 3.3 | 9.7 ± 3.9 |
| stationary | cusum | 0.315 ± 0.260 | 0.288 ± 0.307 | 0.064 | 0.024 | 37.9 ± 7.1 | 38.5 ± 7.1 |
| stationary | robust_z | 0.903 ± 0.030 | 0.897 ± 0.025 | 0.005 | 0.005 | 7.5 ± 2.3 | 8.1 ± 2.5 |
| stationary | variance | 0.527 ± 0.094 | 0.536 ± 0.104 | 0.011 | 0.011 | 30.1 ± 4.8 | 29.7 ± 4.9 |
| stationary | rz_or_var | 0.868 ± 0.048 | 0.875 ± 0.042 | 0.010 | 0.009 | 8.9 ± 2.7 | 8.8 ± 3.7 |
| stationary | random | 0.198 ± 0.064 | 0.198 ± 0.064 | 0.006 | 0.006 | 42.4 ± 3.9 | 42.4 ± 3.9 |
| drift | ewma | 0.721 ± 0.052 | 0.789 ± 0.057 | 0.004 | 0.004 | 13.2 ± 2.2 | 9.2 ± 3.1 |
| drift | cusum | 0.780 ± 0.329 | 0.139 ± 0.206 | 0.868 | 0.015 | 18.2 ± 9.1 | 41.4 ± 5.2 |
| drift | robust_z | 0.869 ± 0.036 | 0.908 ± 0.024 | 0.003 | 0.004 | 10.2 ± 2.6 | 7.4 ± 2.7 |
| drift | variance | 0.545 ± 0.107 | 0.540 ± 0.107 | 0.009 | 0.011 | 29.7 ± 5.0 | 29.7 ± 4.9 |
| drift | rz_or_var | 0.865 ± 0.035 | 0.887 ± 0.044 | 0.009 | 0.009 | 10.1 ± 2.3 | 8.4 ± 3.6 |
| drift | random | 0.198 ± 0.064 | 0.198 ± 0.064 | 0.006 | 0.006 | 42.4 ± 3.9 | 42.4 ± 3.9 |
| drift_missing | ewma | 0.714 ± 0.058 | 0.779 ± 0.060 | 0.005 | 0.004 | 13.2 ± 3.5 | 9.1 ± 3.5 |
| drift_missing | cusum | 0.750 ± 0.272 | 0.155 ± 0.188 | 0.868 | 0.012 | 18.2 ± 9.1 | 41.3 ± 4.3 |
| drift_missing | robust_z | 0.872 ± 0.049 | 0.895 ± 0.023 | 0.004 | 0.005 | 9.7 ± 3.0 | 7.8 ± 2.5 |
| drift_missing | variance | 0.537 ± 0.104 | 0.520 ± 0.109 | 0.011 | 0.011 | 29.9 ± 4.8 | 30.3 ± 5.2 |
| drift_missing | rz_or_var | 0.855 ± 0.028 | 0.851 ± 0.057 | 0.009 | 0.010 | 10.5 ± 1.9 | 10.1 ± 3.6 |
| drift_missing | random | 0.194 ± 0.059 | 0.194 ± 0.059 | 0.006 | 0.006 | 42.5 ± 3.8 | 42.5 ± 3.8 |

### Threshold protocols on the drift scenario, online de-seasonalising (mean over 10 seeds)

| detector | protocol | event F1 | point FAR | cost / 1k |
| --- | --- | ---: | ---: | ---: |
| ewma | static_quantile | 0.789 | 0.0036 | 9.2 |
| ewma | rolling_quantile | 0.708 | 0.0003 | 19.4 |
| ewma | rolling_rescaled | 0.764 | 0.0042 | 9.5 |
| cusum | static_quantile | 0.139 | 0.0151 | 41.4 |
| cusum | rolling_quantile | 0.308 | 0.0137 | 38.6 |
| cusum | rolling_rescaled | 0.376 | 0.0537 | 35.4 |
| robust_z | static_quantile | 0.908 | 0.0037 | 7.4 |
| robust_z | rolling_quantile | 0.625 | 0.0000 | 25.0 |
| robust_z | rolling_rescaled | 0.904 | 0.0039 | 7.6 |
| variance | static_quantile | 0.540 | 0.0105 | 29.7 |
| variance | rolling_quantile | 0.405 | 0.0008 | 35.7 |
| variance | rolling_rescaled | 0.508 | 0.0423 | 31.8 |
| random | static_quantile | 0.198 | 0.0061 | 42.4 |
| random | rolling_quantile | 0.184 | 0.0054 | 42.7 |
| random | rolling_rescaled | 0.195 | 0.0126 | 43.1 |

### Paired per-seed effects (10 seeds; bootstrap 95% CI; exact sign test)

| comparison (b - a) | scenario | detector | metric | mean diff | 95% CI | seeds with diff > 0 | sign p |
| --- | --- | --- | --- | ---: | :---: | ---: | ---: |
| online - fit-once | drift | cusum | event_f1 | -0.641 | [-0.907, -0.327] | 1/10 | 0.021 |
| online - fit-once | drift | ewma | event_f1 | +0.068 | [+0.030, +0.106] | 8/10 | 0.109 |
| online - fit-once | drift | robust_z | event_f1 | +0.039 | [+0.014, +0.068] | 9/10 | 0.021 |
| online - fit-once | stationary | cusum | event_f1 | -0.027 | [-0.103, +0.039] | 2/10 | 0.688 |
| online - fit-once | stationary | ewma | event_f1 | +0.021 | [-0.003, +0.044] | 6/10 | 0.754 |
| online - fit-once | stationary | robust_z | event_f1 | -0.006 | [-0.026, +0.016] | 3/10 | 0.344 |
| online - fit-once | drift | cusum | cost_per_1k | +23.216 | [+14.898, +28.933] | 9/10 | 0.021 |
| online - fit-once | drift | ewma | cost_per_1k | -3.979 | [-5.845, -1.809] | 2/10 | 0.109 |
| online - fit-once | drift | robust_z | cost_per_1k | -2.745 | [-4.699, -1.032] | 1/10 | 0.021 |
| online - fit-once | stationary | cusum | cost_per_1k | +0.636 | [-0.828, +2.161] | 4/10 | 1.000 |
| online - fit-once | stationary | ewma | cost_per_1k | -0.205 | [-1.699, +1.475] | 3/10 | 0.344 |
| online - fit-once | stationary | robust_z | cost_per_1k | +0.608 | [-0.859, +1.957] | 7/10 | 0.344 |
| rescaled - static thr | drift | ewma | event_f1 | -0.025 | [-0.045, -0.008] | 1/10 | 0.039 |
| rescaled - static thr | drift | robust_z | event_f1 | -0.003 | [-0.016, +0.009] | 3/10 | 0.727 |
| 5% missing - none | drift vs drift_missing | ewma | event_f1 | -0.011 | [-0.022, -0.000] | 3/10 | 0.344 |
| 5% missing - none | drift vs drift_missing | robust_z | event_f1 | -0.013 | [-0.023, -0.001] | 1/10 | 0.021 |
| rz_or_var - robust_z | drift | robust_z vs rz_or_var | event_f1 | -0.021 | [-0.042, +0.001] | 3/10 | 0.344 |

### Reading the results

- **H1 supported for the adaptive detectors.** Under drift with the fit-once profile, robust z drops from 0.903 to 0.869 event F1 and EWMA from 0.769 to 0.721. The online de-seasonaliser brings them to 0.908 (+0.039, CI [+0.014, +0.068], 9/10 seeds, p = 0.021) and 0.789 (+0.068, 8/10). Cost per 1k falls from 10.2 to 7.4 and from 13.2 to 9.2. On the stationary stream it changes little (robust z -0.006, EWMA +0.021, both CIs cover 0).
- **CUSUM is the exception, in both directions.** With the fit-once profile and its fit-window scale, drift makes CUSUM alarm on 87% of normal test points. Its event F1 of 0.780 is an artefact of one endless alarm segment touching every event; the cost model charges it 18.2 per 1k for the busy time. The online profile fixes the false alarms (point FAR 0.015) but collapses recall (F1 0.139), because CUSUM's accumulated sums need a stable reference and the adaptive level absorbs part of every shift. Neither configuration is usable under drift.
- **H2 rejected as stated.** The label-free rolling quantile is far too conservative: point FAR 0.0000-0.0008 for EWMA, robust z and the variance detector against a 0.5% target, and robust-z F1 falls to 0.625 (cost 25.0 vs 7.4). The trailing window holds events at about 5% prevalence, 10x the target rate, so its tail rank lands inside the event scores. Rescaling the held-out threshold by a trailing median avoids that, but it does not help either: robust z -0.003 (CI covers 0), EWMA -0.025 (1/10 seeds better), and it doubles the random control's FAR (bounded scores make the ratio touchy). Once the de-seasonaliser adapts, the fixed held-out threshold is already close to target (realised FAR 0.0036-0.0037 for EWMA and robust z).
- **H3 not supported here.** The variance detector alone reaches only 0.52-0.55 F1. Its union with robust z is slightly *worse* than robust z alone under drift (-0.021, CI [-0.042, +0.001]), because robust z already catches every variance burst on this stream (slice-2 recall 1.00) and the union pays twice the false-alarm budget. It is kept as a tool and not recommended.
- **H4 supported.** 5% missing data costs robust z -0.013 F1 (CI [-0.023, -0.001]) and EWMA -0.011 under drift with skipping.

### Test design (slice 3)

- Data: zero drift reproduces slice 2 exactly; the trend and amplitude growth are visible late but not early; gaps hit the target fraction and spare event starts.
- Detectors: skip mode scores gaps 0 and matches strict mode on gap-free data; EWMA and CUSUM scores on observed points equal those of the gap-free series. The variance detector is causal, flags a mean-zero x3.5 burst and ignores a 4-sd level shift that robust z sees. The online profile is causal, tracks a doubling amplitude where the static one leaks it, and a 50-sd spike barely moves later residuals.
- Thresholds: rolling thresholds use only the past and no labels; a drifting score scale breaks the static threshold; the rolling quantile is pushed up by 5% events while the rescaled one stays near target.
- Cost: hand-computed components; the busy term prices an always-on alarm that event metrics call perfect.
- Harness: online residuals are causal and equal static ones on fit; erasing every test label leaves every alarm unchanged; robust z beats random; paired-effect arithmetic by hand. `tests/test_drift_readme_claims.py` re-checks each directional claim above at reduced scale (4 seeds).

## Limits / weaknesses (slice 3)

- Still synthetic, with one drift recipe (linear trend + small random walk + linear amplitude growth). Abrupt regime changes, multiple seasonalities and real incident data are untested.
- The online de-seasonaliser's alpha / gamma / clip were set on a toy series in the unit tests, not tuned on the eval grid. A long level-shift event is partly absorbed into the level at a rate capped by the clip.
- No working CUSUM under drift: it needs a re-referencing scheme (for example, a reset with a fresh robust scale after each alarm), which is not built.
- The rolling protocols assume a trailing window is mostly normal. With 5% event prevalence and a 0.5% target, that fails by construction; a censored or event-aware reference was not tried.
- Cost weights (1 / 10 / 0.1 / 0.02) are assumptions. Rankings between the adaptive detectors held at these weights, but were not tested across a weight grid.
- 10 seeds: sign tests bottom out at p = 0.002 and many effects are 1-3 seeds from a tie.

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

From slices 2-3 (the CUSUM bullet is partly superseded by slice 4):

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
  streaming.py       # causal EWMA / CUSUM / robust-z / variance + static and online seasonal profiles
  event_metrics.py   # point, point-adjusted, event-level metrics, alarm cost
  thresholds.py      # held-out FAR quantile, best-F1, oracle, walk-forward rolling / rescaled
  benchmark.py       # IF vs LOF vs OCSVM tabular benchmark (incl. breast cancer)
  stream_eval.py     # end-to-end streaming harness + Nile CUSUM
  drift_eval.py      # slice 3: drift / missing data x de-seasonalisers x thresholds x costs
  slice4_eval.py     # slice 4: weekly cycle, harmonic RLS, freezable CUSUM, bulk-tail rolling thresholds
scripts/             # run_anomaly_slice.py, run_streaming_slice.py, run_drift_slice.py, run_slice4.py
tests/
```

## License

MIT (demo code).
