"""Slice 3: detectors, de-seasonalisers and threshold protocols under drift and missing data.

Per seed and scenario (:func:`run_drift_experiment`):

1. Draw a stream with :func:`make_drifting_stream`. Scenarios (``SCENARIOS``):
   ``stationary`` (the slice-2 stream), ``drift`` (level trend of 3 marginal
   sd, random-walk baseline, seasonal amplitude +50%) and ``drift_missing``
   (the same plus 5% of points missing in gaps of 5-30).
2. Consecutive windows fit (30%) / cal (20%) / test (50%), as in slice 2.
3. De-seasonalise with the slice-2 fit-once ``SeasonalProfile`` (``static``)
   or the causal ``OnlineSeasonalProfile`` (``online``; initialised on fit,
   then run forward from the start of cal; the fit window keeps its in-sample
   static residual so detector warm-ups see the same history).
4. Score (all ``missing="skip"``): EWMA, CUSUM, robust z, the new explicit
   variance detector and a random control; plus ``rz_or_var``, the union of
   robust-z and variance alarms, each thresholded at half the target rate.
5. Threshold protocols: ``static_quantile`` (slice 2's ``cal_quantile``: the
   FAR rank on label-0 cal points, fixed for the whole test window) and
   ``rolling_quantile`` (walk-forward and label-free: the same rank on the
   trailing ``len(cal)`` observed scores, refreshed every 50 steps from the
   start of test), and ``rolling_rescaled`` (the ``static_quantile``
   threshold multiplied by trailing-median / cal-median score, same window
   and refresh; tracks the score scale without re-estimating the tail).
6. Test-window metrics: event P/R/F1 (tolerance 20), point FAR on observed
   normal points, mean delay, and :func:`alarm_cost` (false page 1, missed
   event 10, 0.1 per step of delay) per 1,000 steps with its components.

Labels touch only the cal-window threshold of ``static_quantile`` and
``rolling_rescaled`` (to drop known anomalies, as in slice 2) and the reported metrics. :func:`paired_effects` reports
per-seed paired differences with an exact sign test and a percentile bootstrap
interval over seeds.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from anomaly_lite.event_metrics import alarm_cost, event_metrics, segments
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.streaming import (
    CUSUMDetector,
    EWMADetector,
    OnlineSeasonalProfile,
    RobustZDetector,
    RollingVarianceDetector,
    SeasonalProfile,
)
from anomaly_lite.thresholds import (
    apply_thresholds,
    false_alarm_threshold,
    rolling_false_alarm_thresholds,
    rolling_rescaled_thresholds,
)

SCENARIOS = {
    "stationary": {"trend_sd": 0.0, "walk_sd": 0.0, "amp_growth": 0.0, "missing_frac": 0.0},
    "drift": {"trend_sd": 3.0, "walk_sd": 0.02, "amp_growth": 0.5, "missing_frac": 0.0},
    "drift_missing": {"trend_sd": 3.0, "walk_sd": 0.02, "amp_growth": 0.5, "missing_frac": 0.05},
}
DESEASON = ("static", "online")
PROTOCOLS = ("static_quantile", "rolling_quantile", "rolling_rescaled")
ROLL_STEP = 50


def residuals(x: np.ndarray, t: np.ndarray, fit: slice, period: int, how: str) -> np.ndarray:
    static = SeasonalProfile(period, missing="skip").fit(x[fit], t[fit])
    r = static.transform(x, t)
    if how == "static":
        return r
    if how != "online":
        raise ValueError(f"unknown de-seasonaliser {how!r}")
    online = OnlineSeasonalProfile(period, missing="skip").fit(x[fit], t[fit])
    out = r.copy()
    out[fit.stop :] = online.transform(x[fit.stop :], t[fit.stop :])
    return out


def score_all(resid: np.ndarray, fit_len: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 777)
    filled = np.where(np.isnan(resid), np.nan, resid)
    return {
        "ewma": EWMADetector(alpha=0.05, clip=3.0, warmup=min(200, fit_len), missing="skip").score(filled),
        "cusum": CUSUMDetector(k=0.5, reference=fit_len, missing="skip").score(filled),
        "robust_z": RobustZDetector(window=200, missing="skip").score(filled),
        "variance": RollingVarianceDetector(short=20, long=400, missing="skip").score(filled),
        "random": np.where(np.isnan(resid), 0.0, rng.uniform(size=resid.size)),
    }


def _alarms(s, y, obs, cal, test, protocol, rate):
    if protocol == "static_quantile":
        keep = obs[cal] & (y[cal] == 0)
        thr = false_alarm_threshold(s[cal][keep], rate)
        return (s[test] > thr).astype(int)
    if protocol == "rolling_quantile":
        thr = rolling_false_alarm_thresholds(s, rate, window=cal.stop - cal.start, start=test.start,
                                             step=ROLL_STEP, valid=obs)
        return apply_thresholds(s, thr)[test]
    if protocol == "rolling_rescaled":
        keep = obs[cal] & (y[cal] == 0)
        base = false_alarm_threshold(s[cal][keep], rate)
        thr = rolling_rescaled_thresholds(s, base, cal, window=cal.stop - cal.start, start=test.start,
                                          step=ROLL_STEP, valid=obs)
        return apply_thresholds(s, thr)[test]
    raise ValueError(f"unknown protocol {protocol!r}")


def _metrics(y_te, obs_te, pred, tolerance):
    ev = event_metrics(y_te, pred, tolerance)
    c = alarm_cost(y_te, pred, tolerance)
    normal = (y_te == 0) & obs_te
    return {"event_f1": ev.f1, "event_precision": ev.precision, "event_recall": ev.recall,
            "mean_delay": ev.mean_delay, "point_far": float(pred[normal].mean()) if normal.any() else np.nan,
            "cost_per_1k": c.per_1k, "n_alarm_points": int(pred.sum()), "n_false": c.n_false, "n_missed": c.n_missed, "delay_steps": c.delay_steps}


def run_drift_experiment(
    seed: int = 0, scenario: str = "drift", n: int = 6000, windows=(0.3, 0.2, 0.5), rate: float = 0.005,
    tolerance: int = 20, n_events: int = 24, deseason: Iterable[str] = DESEASON,
) -> pd.DataFrame:
    stream = make_drifting_stream(n=n, n_events=n_events, warmup=600, min_gap=80, random_state=seed,
                                  **SCENARIOS[scenario])
    fit, cal, test = contiguous_windows(n, windows)
    t = stream.frame["t"].to_numpy()
    x, y = stream.values, stream.labels
    obs = stream.frame["observed"].to_numpy()
    y_te, obs_te = y[test], obs[test]
    rows = []
    for how in deseason:
        resid = residuals(x, t, fit, stream.meta["period"], how)
        scores = score_all(resid, fit.stop - fit.start, seed)
        for proto in PROTOCOLS:
            preds = {d: _alarms(s, y, obs, cal, test, proto, rate) for d, s in scores.items()}
            half = {d: _alarms(scores[d], y, obs, cal, test, proto, rate / 2) for d in ("robust_z", "variance")}
            preds["rz_or_var"] = np.maximum(half["robust_z"], half["variance"])
            for det, pred in preds.items():
                rows.append({"seed": seed, "scenario": scenario, "deseason": how, "protocol": proto,
                             "detector": det, "n_test_events": len(segments(y_te)),
                             **_metrics(y_te, obs_te, pred, tolerance)})
    return pd.DataFrame(rows)


def run_drift_grid(seeds: Iterable[int] = range(10), scenarios: Iterable[str] = tuple(SCENARIOS),
                   **kwargs) -> pd.DataFrame:
    return pd.concat([run_drift_experiment(seed=s, scenario=sc, **kwargs) for sc in scenarios for s in seeds],
                     ignore_index=True)


def summarise(long: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = ["scenario", "deseason", "protocol", "detector"]
    agg = long.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def paired_effects(long: pd.DataFrame, factor: str, a: str, b: str, metric: str,
                   n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per (other keys): mean of (b - a) over seeds, bootstrap 95% CI, wins of b, exact sign-test p."""
    keys = [k for k in ("scenario", "deseason", "protocol", "detector") if k != factor]
    wide = long.pivot_table(index=keys + ["seed"], columns=factor, values=metric).reset_index()
    rng = np.random.default_rng(seed)
    out = []
    for g, sub in wide.groupby(keys, sort=False):
        d = (sub[b] - sub[a]).to_numpy()
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
        nz = d[d != 0]
        p = binomtest(int((nz > 0).sum()), nz.size).pvalue if nz.size else 1.0
        out.append({**dict(zip(keys, g)), "metric": metric, "diff_mean": float(d.mean()),
                    "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
                    "n_b_better": int((d > 0).sum()), "n_ties": int((d == 0).sum()), "n_seeds": int(d.size),
                    "sign_p": float(p)})
    return pd.DataFrame(out)
