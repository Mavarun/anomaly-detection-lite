"""Slice 9: does calibrating the static threshold on fit + calibration windows (2.5x the points) help under t(3)?

Slice 6 traced part of the t(3) loss to the static threshold: the 0.5% rank of the 1,200-point
calibration window rests on about 6 points. Slice 7 tried a GPD tail fit (OFF); slice 8 showed a
static threshold can reach the guardrail in principle (oracle 0.795 vs line 0.741). The simplest
way to put more points under the 0.5% rank is to pool the normal points of the fit window (1,800)
with the calibration window (1,200): about 15 points instead of 6. The catch is that fit-window
scores are in-sample (the de-seasonaliser and the detectors' reference statistics are fitted
there), so they may be too tight and pull the threshold down.

``pooled_calibration`` is that rule. It is compared with the shipped ``static_quantile`` on the
slice 8 streams (``SCENARIOS6``, ``harm_dw3``) for robust_z, cusum_k15 and cusum_pw.

Seeds: evaluation 500-509, fresh; dev seed 150 only checks that the harness runs.
:data:`PREREGISTERED` was committed before any evaluation seed ran.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from anomaly_lite.drift_eval import _metrics
from anomaly_lite.slice5_eval import residuals5
from anomaly_lite.slice6_eval import SCENARIOS6, score6
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.thresholds import false_alarm_threshold

DETECTORS9 = ("robust_z", "cusum_k15", "cusum_pw")
EVAL_SEEDS = tuple(range(500, 510))
DEV_SEEDS = (150,)
RATE = 0.005
FAR_CAP = 0.0075  # 1.5x the 0.5% budget

PREREGISTERED = {
    "H1_pooled_calibration": (
        "The shipped robust_z static threshold switches to pooled (fit + calibration normal points) calibration iff, "
        "on 'weekend_t3' over seeds 500-509, mean event-F1(pooled - quantile) > 0 with pooled better on >= 7 of 10 "
        "seeds, AND on Gaussian 'weekend' mean event-F1(pooled - quantile) >= -0.01, AND pooled's mean realised "
        "false-alarm rate on normal test points of 'weekend_t3' is <= 0.0075 (1.5x the 0.5% budget). Otherwise it "
        "ships OFF."
    ),
    "H2_guardrail": (
        "Under the rule H1 picks, robust_z mean F1 on 'weekend_t3' >= mean F1 on 'weekend' - 0.10 (slice 6's "
        "guardrail). Reported; a failure ships nothing."
    ),
    "expectation": (
        "Written before the run: H1 fails on the FAR cap, because in-sample fit-window scores are tighter than "
        "held-out ones and pull the pooled threshold down."
    ),
    "reported_not_decided": "weekend_t5 and the two CUSUMs; thresholds and realised FARs for both rules.",
}


def run_slice9_experiment(seed: int, scenario: str, n: int = 6000, windows=(0.3, 0.2, 0.5), rate: float = RATE,
                          tolerance: int = 20, n_events: int = 24) -> pd.DataFrame:
    stream = make_drifting_stream(n=n, n_events=n_events, warmup=600, min_gap=80, random_state=seed,
                                  **SCENARIOS6[scenario])
    fit, cal, test = contiguous_windows(n, windows)
    t, x, y = stream.frame["t"].to_numpy(), stream.values, stream.labels
    obs = stream.frame["observed"].to_numpy()
    resid = residuals5(x, t, fit, stream.meta["period"], "harm_dw3")
    keep_cal = obs[cal] & (y[cal] == 0)
    keep_fit = obs[fit] & (y[fit] == 0)
    normal_test = obs[test] & (y[test] == 0)
    rows = []
    for det, s in score6(resid, fit.stop - fit.start).items():
        if det not in DETECTORS9:
            continue
        cal_s = s[cal][keep_cal]
        cal_s = cal_s[np.isfinite(cal_s)]
        fit_s = s[fit][keep_fit]
        fit_s = fit_s[np.isfinite(fit_s)]
        thr = {"quantile": false_alarm_threshold(cal_s, rate),
               "pooled": false_alarm_threshold(np.r_[fit_s, cal_s], rate)}
        row = {"seed": seed, "scenario": scenario, "detector": det, "n_cal": int(cal_s.size),
               "n_pooled": int(cal_s.size + fit_s.size)}
        for rule, th in thr.items():
            m = _metrics(y[test], obs[test], (s[test] > th).astype(int), tolerance)
            row[f"{rule}_threshold"] = float(th)
            row[f"{rule}_f1"] = float(m["event_f1"])
            row[f"{rule}_far"] = float((s[test][normal_test] > th).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_slice9_grid(seeds: Iterable[int] = EVAL_SEEDS, scenarios: Iterable[str] = tuple(SCENARIOS6)) -> pd.DataFrame:
    return pd.concat([run_slice9_experiment(s, sc) for sc in scenarios for s in seeds], ignore_index=True)


def decide(df: pd.DataFrame) -> dict:
    rz = df[df.detector == "robust_z"].set_index(["scenario", "seed"])
    t3, g = rz.loc["weekend_t3"], rz.loc["weekend"]
    d3 = (t3.pooled_f1 - t3.quantile_f1).to_numpy()
    dg = (g.pooled_f1 - g.quantile_f1).to_numpy()
    far = float(t3.pooled_far.mean())
    h1 = bool(d3.mean() > 0 and int((d3 > 0).sum()) >= 7 and dg.mean() >= -0.01 and far <= FAR_CAP)
    rule = "pooled" if h1 else "quantile"
    t3_f1, g_f1 = float(t3[f"{rule}_f1"].mean()), float(g[f"{rule}_f1"].mean())
    return {"H1": h1, "t3_f1_diff": float(d3.mean()), "t3_seeds_better": int((d3 > 0).sum()),
            "gauss_f1_diff": float(dg.mean()), "t3_pooled_far": far, "t3_quantile_far": float(t3.quantile_far.mean()),
            "rule": rule, "t3_f1": t3_f1, "gauss_f1": g_f1, "guardrail_margin": t3_f1 - (g_f1 - 0.10),
            "H2": bool(t3_f1 >= g_f1 - 0.10)}


def posthoc_fit_window(seeds: Iterable[int] = EVAL_SEEDS, scenarios: Iterable[str] = ("weekend", "weekend_t3"),
                       n: int = 6000, windows=(0.3, 0.2, 0.5), rate: float = RATE, head: int = 600) -> pd.DataFrame:
    """POST-HOC (after the decision; decides nothing): why pooling raised the thresholds.

    Per seed, scenario and detector: the share of normal fit-window scores above the held-out
    calibration threshold (nominally ``rate``), and the share of those exceedances in the first
    ``head`` points of the fit window (``head / fit length`` if they were spread evenly)."""
    out = []
    for sc in scenarios:
        for seed in seeds:
            stream = make_drifting_stream(n=n, n_events=24, warmup=600, min_gap=80, random_state=int(seed),
                                          **SCENARIOS6[sc])
            fit, cal, _ = contiguous_windows(n, windows)
            t, x, y = stream.frame["t"].to_numpy(), stream.values, stream.labels
            obs = stream.frame["observed"].to_numpy()
            scores = score6(residuals5(x, t, fit, stream.meta["period"], "harm_dw3"), fit.stop - fit.start)
            for det in DETECTORS9:
                s = scores[det]
                c = s[cal][obs[cal] & (y[cal] == 0)]
                thr = false_alarm_threshold(c[np.isfinite(c)], rate)
                f = s[fit]
                k = obs[fit] & (y[fit] == 0) & np.isfinite(f)
                pos = np.arange(fit.start, fit.stop)[k]
                exc = pos[f[k] > thr]
                out.append({"seed": int(seed), "scenario": sc, "detector": det,
                            "fit_exceed_rate": float((f[k] > thr).mean()),
                            "head_share": float((exc < fit.start + head).mean()) if exc.size else float("nan"),
                            "head_share_if_even": head / (fit.stop - fit.start)})
    return pd.DataFrame(out)
