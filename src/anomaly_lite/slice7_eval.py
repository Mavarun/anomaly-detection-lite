"""Slice 7: a heavy-tail-aware static threshold (peaks over threshold, GPD).

Slice 6 found that t(3) noise pushes every static threshold up (robust z 2.89 -> 4.52): the
empirical 0.5% rank of a 1,200-point calibration window rests on about 6 tail points. This slice
swaps only the threshold rule: ``static_gpd`` fits a generalized Pareto to the top 10% of the
normal calibration scores and extrapolates to the 0.5% rate (:func:`gpd_tail_threshold`). Same
streams (``SCENARIOS6``), de-seasonaliser (``harm_dw3``) and detectors as slice 6.

Seeds: evaluation 300-309, fresh (slices 2-6 used 0-9 and 100-105). :data:`PREREGISTERED` was
committed before any evaluation seed was run; dev seeds 100-102 only check that the harness runs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from anomaly_lite.drift_eval import _metrics, paired_effects
from anomaly_lite.event_metrics import segments
from anomaly_lite.slice5_eval import residuals5
from anomaly_lite.slice6_eval import SCENARIOS6, score6
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.thresholds import false_alarm_threshold, gpd_tail_threshold

PROTOCOLS7 = ("static_quantile", "static_gpd")
DETECTORS7 = ("robust_z", "cusum_k15", "cusum_pw")
EVAL_SEEDS = tuple(range(300, 310))
DEV_SEEDS = (100, 101, 102)

PREREGISTERED = {
    "H1_gpd_for_robust_z": (
        "The recommended static threshold for the shipped robust_z + harm_dw3 becomes static_gpd iff, on "
        "evaluation seeds 300-309: mean event-F1(static_gpd - static_quantile) > 0 on 'weekend_t3' with gpd better "
        "on >= 7 of 10 seeds, AND >= -0.01 on Gaussian 'weekend', AND mean cost_per_1k on 'weekend_t3' is no higher. "
        "Otherwise static_quantile stays."
    ),
    "H2_guardrail_with_the_chosen_rule": (
        "With the rule H1 picks, robust_z's mean event F1 on 'weekend_t3' is >= its Gaussian 'weekend' F1 - 0.10 "
        "(slice 6's guardrail, failed there at -0.120). A failure ships nothing more and stays a weakness."
    ),
    "reported_not_decided": "weekend_t5, the CUSUMs under both rules, the thresholds and the realised FAR.",
}


def threshold7(cal_scores, protocol: str, rate: float = 0.005) -> float:
    if protocol == "static_quantile":
        return false_alarm_threshold(cal_scores, rate)
    if protocol == "static_gpd":
        return gpd_tail_threshold(cal_scores, rate)
    raise ValueError(protocol)


def run_slice7_experiment(seed: int, scenario: str, n: int = 6000, windows=(0.3, 0.2, 0.5), rate: float = 0.005,
                          tolerance: int = 20, n_events: int = 24) -> pd.DataFrame:
    stream = make_drifting_stream(n=n, n_events=n_events, warmup=600, min_gap=80, random_state=seed,
                                  **SCENARIOS6[scenario])
    fit, cal, test = contiguous_windows(n, windows)
    t, x, y = stream.frame["t"].to_numpy(), stream.values, stream.labels
    obs = stream.frame["observed"].to_numpy()
    resid = residuals5(x, t, fit, stream.meta["period"], "harm_dw3")
    keep = obs[cal] & (y[cal] == 0)
    rows = []
    for det, s in score6(resid, fit.stop - fit.start).items():
        if det not in DETECTORS7:
            continue
        cal_s = s[cal][keep]
        cal_s = cal_s[np.isfinite(cal_s)]
        normal_test = obs[test] & (y[test] == 0)
        for proto in PROTOCOLS7:
            thr = threshold7(cal_s, proto, rate)
            pred = (s[test] > thr).astype(int)
            far = float((s[test][normal_test] > thr).mean())
            rows.append({"seed": seed, "scenario": scenario, "deseason": "harm_dw3", "protocol": proto,
                         "detector": det, "threshold": float(thr), "realised_far": far,
                         "n_test_events": len(segments(y[test])), **_metrics(y[test], obs[test], pred, tolerance)})
    return pd.DataFrame(rows)


def run_slice7_grid(seeds: Iterable[int] = EVAL_SEEDS, scenarios: Iterable[str] = tuple(SCENARIOS6)) -> pd.DataFrame:
    return pd.concat([run_slice7_experiment(s, sc) for sc in scenarios for s in seeds], ignore_index=True)


def _gpd_effect(long, scenario, metric):
    sub = long[(long.scenario == scenario) & (long.detector == "robust_z")]
    e = paired_effects(sub, "protocol", "static_quantile", "static_gpd", metric)
    assert len(e) == 1, e
    return e.iloc[0]


def decide(long: pd.DataFrame) -> dict:
    t3, g = _gpd_effect(long, "weekend_t3", "event_f1"), _gpd_effect(long, "weekend", "event_f1")
    rz = long[long.detector == "robust_z"]
    cost = rz[rz.scenario == "weekend_t3"].groupby("protocol").cost_per_1k.mean()
    parts = {"t3_gain": bool(t3.diff_mean > 0 and t3.n_b_better >= 7), "gauss_ok": bool(g.diff_mean >= -0.01),
             "cost_ok": bool(cost["static_gpd"] <= cost["static_quantile"])}
    chosen = "static_gpd" if all(parts.values()) else "static_quantile"
    f1 = rz[rz.protocol == chosen].groupby("scenario").event_f1.mean()
    return {"H1": all(parts.values()), "H1_parts": parts, "H1_t3": t3.to_dict(), "H1_gauss": g.to_dict(),
            "chosen": chosen, "H2": bool(f1["weekend_t3"] >= f1["weekend"] - 0.10),
            "robust_z_f1_chosen": {k: float(v) for k, v in f1.items()}}
