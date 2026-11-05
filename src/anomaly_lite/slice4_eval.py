"""Slice 4: a second seasonality, harmonic RLS de-seasonalising, a freezable CUSUM and contamination-robust rolling thresholds.

Per seed and scenario (:func:`run_slice4_experiment`), reusing the slice-3 stream, windows
(fit 30% / cal 20% / test 50%), metrics, alarm cost and paired statistics:

* Scenarios (``SCENARIOS4``): ``drift`` (slice 3), ``weekly`` (drift plus a 672-step
  sinusoid of amplitude 1.5, a weekly cycle over the 96-step day) and
  ``weekly_missing`` (the same plus 5% missing in gaps).
* De-seasonalisers: ``static`` (slice-2 per-phase median, fit once), ``online``
  (slice-3 Holt-Winters, one period), ``harm_d`` (:class:`OnlineHarmonicProfile`, period 96)
  and ``harm_dw`` (periods 96 and 672). All see the fit window only to initialise, then run forward.
* Detectors: EWMA, robust z, the slice-2 fixed-reference ``cusum`` (allowance k = 0.5, in sd of the
  fit-window residual), the same with k = 1.5 (``cusum_k15``), the new ``cusum_frozen`` /
  ``cusum_frozen_k15`` (:class:`FrozenReferenceCUSUM`, k = 0.5 / 1.5) and the slice-3 ``variance``
  detector. Why k = 1.5: the stream noise is AR(1) with phi = 0.6, so a sum of residuals has about
  twice the sd of an i.i.d. sum, and the textbook k = 0.5 (tuned for i.i.d. noise) is far too
  twitchy; k was tried at 0.5 / 1.0 / 1.5 on the development seeds only.
* Threshold protocols at a 0.5% target false-alarm rate: ``static_quantile`` (label-free except for
  dropping known anomalies on cal), ``rolling_rescaled`` (slice 3) and ``rolling_bulk`` (label-free
  walk-forward, :func:`rolling_bulk_tail_thresholds`; half-normal tail for the |z|-type scores EWMA and
  robust z, exponential for the accumulated or log-ratio scores CUSUM and variance, a rule fixed
  before the evaluation seeds were run).

Every number is on the test window. Hyper-parameters (forgetting 0.995, the 80/90% quantile anchors,
freeze level 2) were set on development seeds 100-105, never on the evaluation seeds 0-9.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from anomaly_lite.drift_eval import SCENARIOS as DRIFT_SCENARIOS
from anomaly_lite.drift_eval import ROLL_STEP, _metrics, paired_effects, summarise  # noqa: F401
from anomaly_lite.event_metrics import segments
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.streaming import (
    CUSUMDetector,
    EWMADetector,
    FrozenReferenceCUSUM,
    OnlineHarmonicProfile,
    OnlineSeasonalProfile,
    RobustZDetector,
    RollingVarianceDetector,
    SeasonalProfile,
)
from anomaly_lite.thresholds import (
    apply_thresholds,
    false_alarm_threshold,
    rolling_bulk_tail_thresholds,
    rolling_rescaled_thresholds,
)

WEEK = 7 * 96
SCENARIOS4 = {
    "drift": dict(DRIFT_SCENARIOS["drift"]),
    "weekly": {**DRIFT_SCENARIOS["drift"], "season2_period": WEEK, "season2_amp": 1.5},
    "weekly_missing": {**DRIFT_SCENARIOS["drift_missing"], "season2_period": WEEK, "season2_amp": 1.5},
}
DESEASON4 = ("static", "online", "harm_d", "harm_dw")
DETECTORS4 = ("ewma", "robust_z", "cusum", "cusum_k15", "cusum_frozen", "cusum_frozen_k15", "variance")
PROTOCOLS4 = ("static_quantile", "rolling_rescaled", "rolling_bulk")
TAIL = {"ewma": "halfnormal", "robust_z": "halfnormal", "cusum": "exponential", "cusum_k15": "exponential",
        "cusum_frozen": "exponential", "cusum_frozen_k15": "exponential", "variance": "exponential"}


def residuals4(x: np.ndarray, t: np.ndarray, fit: slice, period: int, how: str, forgetting: float = 0.995) -> np.ndarray:
    """Residual series: in-sample static residual on the fit window, causal forward run after it."""
    static = SeasonalProfile(period, missing="skip").fit(x[fit], t[fit])
    out = static.transform(x, t)
    if how == "static":
        return out
    rest = slice(fit.stop, None)
    if how == "online":
        out[rest] = OnlineSeasonalProfile(period, missing="skip").fit(x[fit], t[fit]).transform(x[rest], t[rest])
    elif how in ("harm_d", "harm_dw"):
        periods = (period,) if how == "harm_d" else (period, WEEK)
        out[rest] = OnlineHarmonicProfile(periods, 1, forgetting, missing="skip").fit(x[fit], t[fit]).transform(
            x[rest], t[rest])
    else:
        raise ValueError(f"unknown de-seasonaliser {how!r}")
    return out


def score4(resid: np.ndarray, fit_len: int) -> dict[str, np.ndarray]:
    return {
        "ewma": EWMADetector(alpha=0.05, clip=3.0, warmup=min(200, fit_len), missing="skip").score(resid),
        "robust_z": RobustZDetector(window=200, missing="skip").score(resid),
        "cusum": CUSUMDetector(k=0.5, reference=fit_len, missing="skip").score(resid),
        "cusum_k15": CUSUMDetector(k=1.5, reference=fit_len, missing="skip").score(resid),
        "cusum_frozen": FrozenReferenceCUSUM(k=0.5, reference=fit_len, missing="skip").score(resid),
        "cusum_frozen_k15": FrozenReferenceCUSUM(k=1.5, reference=fit_len, missing="skip").score(resid),
        "variance": RollingVarianceDetector(short=20, long=400, missing="skip").score(resid),
    }


def alarms4(s, y, obs, cal, test, protocol, det, rate=0.005):
    keep = obs[cal] & (y[cal] == 0)
    window = cal.stop - cal.start
    if protocol == "static_quantile":
        return (s[test] > false_alarm_threshold(s[cal][keep], rate)).astype(int)
    if protocol == "rolling_rescaled":
        base = false_alarm_threshold(s[cal][keep], rate)
        thr = rolling_rescaled_thresholds(s, base, cal, window=window, start=test.start, step=ROLL_STEP, valid=obs)
        return apply_thresholds(s, thr)[test]
    if protocol == "rolling_bulk":
        thr = rolling_bulk_tail_thresholds(s, rate, window=window, start=test.start, step=ROLL_STEP, valid=obs,
                                           tail=TAIL[det])
        return apply_thresholds(s, thr)[test]
    raise ValueError(f"unknown protocol {protocol!r}")


def run_slice4_experiment(seed: int = 0, scenario: str = "weekly", n: int = 6000, windows=(0.3, 0.2, 0.5),
                          rate: float = 0.005, tolerance: int = 20, n_events: int = 24,
                          deseason: Iterable[str] = DESEASON4, protocols: Iterable[str] = PROTOCOLS4,
                          forgetting: float = 0.995) -> pd.DataFrame:
    stream = make_drifting_stream(n=n, n_events=n_events, warmup=600, min_gap=80, random_state=seed,
                                  **SCENARIOS4[scenario])
    fit, cal, test = contiguous_windows(n, windows)
    t = stream.frame["t"].to_numpy()
    x, y = stream.values, stream.labels
    obs = stream.frame["observed"].to_numpy()
    y_te, obs_te = y[test], obs[test]
    rows = []
    for how in deseason:
        resid = residuals4(x, t, fit, stream.meta["period"], how, forgetting)
        scores = score4(resid, fit.stop - fit.start)
        for proto in protocols:
            for det, s in scores.items():
                pred = alarms4(s, y, obs, cal, test, proto, det, rate)
                rows.append({"seed": seed, "scenario": scenario, "deseason": how, "protocol": proto,
                             "detector": det, "forgetting": forgetting, "n_test_events": len(segments(y_te)),
                             **_metrics(y_te, obs_te, pred, tolerance)})
    return pd.DataFrame(rows)


def run_slice4_grid(seeds: Iterable[int] = range(10), scenarios: Iterable[str] = tuple(SCENARIOS4),
                    **kwargs) -> pd.DataFrame:
    return pd.concat([run_slice4_experiment(seed=s, scenario=sc, **kwargs) for sc in scenarios for s in seeds],
                     ignore_index=True)
