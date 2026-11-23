"""Slice 11: a longer memory for the weekly de-seasonaliser's online update.

``weekly_harmonic_profile`` runs recursive least squares with ``forgetting = 0.995``, a memory of
about 200 steps, while the weekly cycle it tracks is 672 steps long. With that memory the three
weekly harmonics are re-estimated from under a third of a week, so heavy-tailed noise can pull
the weekend shape around. Slice 10's guardrail (H2) failed by 0.002 on t(3) noise. This slice
asks whether ``forgetting = 0.999`` (memory ~1,000 steps, about 1.5 weeks) is the cheaper fix.

Only the residual changes; robust_z and the shipped calibration-window quantile threshold stay.
Seeds: evaluation 700-709, fresh. Dev seeds 150-152 were run once to write the expectation (t3
F1 0.723/0.685/0.705 -> 0.735/0.729/0.640, Gaussian 0.870/0.917/0.873 -> 0.845/0.948/0.850).
:data:`PREREGISTERED` was committed before any evaluation seed ran.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from anomaly_lite.drift_eval import _metrics
from anomaly_lite.slice5_eval import residuals5
from anomaly_lite.slice6_eval import SCENARIOS6
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.streaming import RobustZDetector
from anomaly_lite.thresholds import false_alarm_threshold

EVAL_SEEDS = tuple(range(700, 710))
DEV_SEEDS = (150, 151, 152)
ARMS = {"f995": 0.995, "f999": 0.999}
RATE = 0.005
FAR_CAP = 0.0075

PREREGISTERED = {
    "H1_forgetting": (
        "The weekly de-seasonaliser's default forgetting switches 0.995 -> 0.999 iff, with robust_z and the "
        "calibration-window quantile threshold over seeds 700-709: on 'weekend_t3' mean event-F1(f999 - f995) > 0 "
        "with f999 better on >= 7 of 10 seeds, AND on Gaussian 'weekend' mean event-F1(f999 - f995) >= -0.01, AND "
        "f999's mean false-alarm rate on normal 'weekend_t3' test points is <= 0.0075."
    ),
    "H2_guardrail": "Under the forgetting H1 picks, robust_z mean F1 on 'weekend_t3' >= mean F1 on 'weekend' - 0.10.",
    "expectation": (
        "Written before the run, from dev seeds 150-152 (t3 better on 2 of 3, mean +0.000; Gaussian -0.006): "
        "H1 fails on the 7-of-10 t3 part; H2 fails as in slice 10."
    ),
    "reported_not_decided": "weekend_t5, and each arm's threshold and FAR.",
}


def run_slice11_experiment(seed: int, scenario: str, n: int = 6000, windows=(0.3, 0.2, 0.5), rate: float = RATE,
                           tolerance: int = 20, n_events: int = 24) -> dict:
    stream = make_drifting_stream(n=n, n_events=n_events, warmup=600, min_gap=80, random_state=seed,
                                  **SCENARIOS6[scenario])
    fit, cal, test = contiguous_windows(n, windows)
    t, x, y = stream.frame["t"].to_numpy(), stream.values, stream.labels
    obs = stream.frame["observed"].to_numpy()
    keep_cal, normal_test = obs[cal] & (y[cal] == 0), obs[test] & (y[test] == 0)
    row = {"seed": int(seed), "scenario": scenario}
    for arm, f in ARMS.items():
        s = RobustZDetector(window=200, missing="skip").score(residuals5(x, t, fit, stream.meta["period"], "harm_dw3", f))
        c = s[cal][keep_cal]
        th = false_alarm_threshold(c[np.isfinite(c)], rate)
        m = _metrics(y[test], obs[test], (s[test] > th).astype(int), tolerance)
        row[f"{arm}_threshold"] = float(th)
        row[f"{arm}_f1"] = float(m["event_f1"])
        row[f"{arm}_far"] = float((s[test][normal_test] > th).mean())
    return row


def run_slice11_grid(seeds: Iterable[int] = EVAL_SEEDS, scenarios: Iterable[str] = tuple(SCENARIOS6)) -> pd.DataFrame:
    return pd.DataFrame([run_slice11_experiment(s, sc) for sc in scenarios for s in seeds])


def decide(df: pd.DataFrame) -> dict:
    d = df.set_index(["scenario", "seed"])
    t3, g = d.loc["weekend_t3"], d.loc["weekend"]
    d3, dg = (t3.f999_f1 - t3.f995_f1).to_numpy(), (g.f999_f1 - g.f995_f1).to_numpy()
    far = float(t3.f999_far.mean())
    h1 = bool(d3.mean() > 0 and int((d3 > 0).sum()) >= 7 and dg.mean() >= -0.01 and far <= FAR_CAP)
    arm = "f999" if h1 else "f995"
    t3_f1, g_f1 = float(t3[f"{arm}_f1"].mean()), float(g[f"{arm}_f1"].mean())
    return {"H1": h1, "t3_f1_diff": float(d3.mean()), "t3_seeds_better": int((d3 > 0).sum()),
            "gauss_f1_diff": float(dg.mean()), "gauss_seeds_better": int((dg > 0).sum()),
            "t3_f999_far": far, "t3_f995_far": float(t3.f995_far.mean()), "arm": arm,
            "t3_f1": t3_f1, "gauss_f1": g_f1, "guardrail_margin": t3_f1 - (g_f1 - 0.10), "H2": bool(t3_f1 >= g_f1 - 0.10)}
