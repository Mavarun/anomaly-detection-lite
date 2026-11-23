"""Slice 10: pool fit-window scores only after the same weekly de-seasonaliser the test window sees.

Slice 9's ``pooled`` rule failed, and its post-hoc reading found fit-window normal scores far
wider than held-out ones (exceed rates 3.5-36% against a nominal 0.5%). The cause is in
``residuals5``: with ``harm_dw3`` the fit window keeps the slice-4 *daily-only* static residual,
while every later point gets the weekly harmonic profile. The fit-window residual therefore still
holds the weekend dip, so pooling mixed two different residual series. (Dev seeds 150-152 show
the robust_z exceedances cycling with the 672-bar week, not crowding the head.)

``weekly_fit_scores`` re-scores the fit window on a residual series whose fit window is the
weekly profile run in-sample (fitted on the fit window, then transformed over it); cal and test
scores are untouched. ``pooled_weekly`` calibrates on those fit-window normal scores, minus the
first ``WARMUP`` points (robust_z's 200-point window), plus the calibration window.

Seeds: evaluation 600-609, fresh; dev seeds 150-152 only. :data:`PREREGISTERED` was committed
before any evaluation seed ran.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from anomaly_lite.drift_eval import _metrics
from anomaly_lite.slice4_eval import WEEK
from anomaly_lite.slice5_eval import residuals5
from anomaly_lite.slice6_eval import SCENARIOS6, score6
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.streaming import weekly_harmonic_profile
from anomaly_lite.thresholds import false_alarm_threshold

DETECTORS10 = ("robust_z", "cusum_k15", "cusum_pw")
EVAL_SEEDS = tuple(range(600, 610))
DEV_SEEDS = (150, 151, 152)
RATE = 0.005
FAR_CAP = 0.0075
WARMUP = 200
MECHANISM_CAP = 0.01

PREREGISTERED = {
    "C0_mechanism": (
        "Reported, decides nothing about shipping: with the weekly de-seasonaliser in the fit window, robust_z's "
        "share of fit-window normal scores (after the 200-point warm-up) above the held-out calibration threshold "
        "is <= 0.01 on both 'weekend' and 'weekend_t3' (slice 9: 0.066 and 0.035)."
    ),
    "H1_pooled_weekly": (
        "The shipped robust_z static threshold switches to pooled_weekly iff, on 'weekend_t3' over seeds 600-609, "
        "mean event-F1(pooled_weekly - quantile) > 0 with pooled_weekly better on >= 7 of 10 seeds, AND on "
        "Gaussian 'weekend' mean event-F1(pooled_weekly - quantile) >= -0.01, AND pooled_weekly's mean realised "
        "false-alarm rate on normal test points of 'weekend_t3' is <= 0.0075 (slice 9's bar, unchanged)."
    ),
    "H2_guardrail": "Under the rule H1 picks, robust_z mean F1 on 'weekend_t3' >= mean F1 on 'weekend' - 0.10.",
    "expectation": (
        "Written before the run, from dev seeds 150-152: C0 holds; H1 fails on the 7-of-10 t3 seeds part (dev t3 "
        "gains were split), while the Gaussian part passes; H2 fails."
    ),
    "reported_not_decided": "weekend_t5 and the two CUSUMs, whose fit-window scores stay wide (CUSUM state).",
}


def weekly_fit_scores(x, t, fit: slice, period: int) -> dict[str, np.ndarray]:
    """Detector scores on a residual series whose fit window uses the weekly profile in-sample."""
    r = residuals5(x, t, fit, period, "harm_dw3")
    r[fit] = weekly_harmonic_profile(period, WEEK).fit(x[fit], t[fit]).transform(x[fit], t[fit])
    return score6(r, fit.stop - fit.start)


def run_slice10_experiment(seed: int, scenario: str, n: int = 6000, windows=(0.3, 0.2, 0.5), rate: float = RATE,
                           tolerance: int = 20, n_events: int = 24) -> pd.DataFrame:
    stream = make_drifting_stream(n=n, n_events=n_events, warmup=600, min_gap=80, random_state=seed,
                                  **SCENARIOS6[scenario])
    fit, cal, test = contiguous_windows(n, windows)
    t, x, y = stream.frame["t"].to_numpy(), stream.values, stream.labels
    obs = stream.frame["observed"].to_numpy()
    period = stream.meta["period"]
    shipped = score6(residuals5(x, t, fit, period, "harm_dw3"), fit.stop - fit.start)
    weekly = weekly_fit_scores(x, t, fit, period)
    keep_cal = obs[cal] & (y[cal] == 0)
    normal_test = obs[test] & (y[test] == 0)
    keep_fit = obs[fit] & (y[fit] == 0)
    keep_fit[:WARMUP] = False
    rows = []
    for det in DETECTORS10:
        s = shipped[det]
        cal_s = s[cal][keep_cal]
        cal_s = cal_s[np.isfinite(cal_s)]
        f = weekly[det][fit]
        fit_s = f[keep_fit & np.isfinite(f)]
        q = false_alarm_threshold(cal_s, rate)
        thr = {"quantile": q, "pooled_weekly": false_alarm_threshold(np.r_[fit_s, cal_s], rate)}
        row = {"seed": int(seed), "scenario": scenario, "detector": det,
               "fit_exceed_rate": float((fit_s > q).mean()), "n_pooled": int(fit_s.size + cal_s.size)}
        for rule, th in thr.items():
            m = _metrics(y[test], obs[test], (s[test] > th).astype(int), tolerance)
            row[f"{rule}_threshold"] = float(th)
            row[f"{rule}_f1"] = float(m["event_f1"])
            row[f"{rule}_far"] = float((s[test][normal_test] > th).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_slice10_grid(seeds: Iterable[int] = EVAL_SEEDS, scenarios: Iterable[str] = tuple(SCENARIOS6)) -> pd.DataFrame:
    return pd.concat([run_slice10_experiment(s, sc) for sc in scenarios for s in seeds], ignore_index=True)


def decide(df: pd.DataFrame) -> dict:
    rz = df[df.detector == "robust_z"].set_index(["scenario", "seed"])
    t3, g = rz.loc["weekend_t3"], rz.loc["weekend"]
    d3 = (t3.pooled_weekly_f1 - t3.quantile_f1).to_numpy()
    dg = (g.pooled_weekly_f1 - g.quantile_f1).to_numpy()
    far = float(t3.pooled_weekly_far.mean())
    c0 = {sc: float(rz.loc[sc].fit_exceed_rate.mean()) for sc in ("weekend", "weekend_t3")}
    h1 = bool(d3.mean() > 0 and int((d3 > 0).sum()) >= 7 and dg.mean() >= -0.01 and far <= FAR_CAP)
    rule = "pooled_weekly" if h1 else "quantile"
    t3_f1, g_f1 = float(t3[f"{rule}_f1"].mean()), float(g[f"{rule}_f1"].mean())
    return {"C0": bool(all(v <= MECHANISM_CAP for v in c0.values())), "C0_exceed": c0,
            "H1": h1, "t3_f1_diff": float(d3.mean()), "t3_seeds_better": int((d3 > 0).sum()),
            "gauss_f1_diff": float(dg.mean()), "gauss_seeds_better": int((dg > 0).sum()),
            "t3_pooled_weekly_far": far, "t3_quantile_far": float(t3.quantile_far.mean()),
            "rule": rule, "t3_f1": t3_f1, "gauss_f1": g_f1, "guardrail_margin": t3_f1 - (g_f1 - 0.10),
            "H2": bool(t3_f1 >= g_f1 - 0.10)}
