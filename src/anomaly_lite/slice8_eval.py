"""Slice 8: is slice 6's heavy-tail guardrail reachable by any static threshold?

Slices 6 and 7 both failed the guardrail "robust_z F1 under t(3) noise >= Gaussian F1 - 0.10"
(-0.120 and -0.103). Slice 7's post-hoc said the t(3) gap is the true tail quantile, not
estimation noise, and suggested a rank or log transform next. That suggestion was wrong for this
setting: a monotone transform of the score with a static threshold flags exactly the same
points, so it cannot change F1. The question worth asking first is whether *any* static
threshold reaches the guardrail. ``oracle_threshold`` picks, per seed, the threshold that
maximises event F1 on the test window itself, using the labels. It is an upper bound, not a
method.

Seeds: evaluation 400-409, fresh. :data:`PREREGISTERED` was committed before any of them ran.
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

DETECTORS8 = ("robust_z", "cusum_k15", "cusum_pw")
EVAL_SEEDS = tuple(range(400, 410))
QUANTILE_GRID = tuple(np.round(np.r_[np.linspace(0.90, 0.99, 46), np.linspace(0.9905, 0.9995, 19)], 6))

PREREGISTERED = {
    "H1_guardrail_reachable": (
        "On seeds 400-409, robust_z + harm_dw3 with the per-seed oracle threshold reaches mean event F1 on "
        "'weekend_t3' >= (mean F1 of the shipped static_quantile on Gaussian 'weekend') - 0.10. If not, no static "
        "threshold can pass the guardrail for this detector, and threshold work stops being the route to it."
    ),
    "H2_headroom": (
        "The oracle's mean t3 F1 gain over static_quantile is reported with a paired bootstrap CI; if the CI's upper "
        "end is below 0.05, threshold choice is recorded as exhausted for t3 (any better static rule gains < 0.05)."
    ),
    "reported_not_decided": "weekend_t5 and the CUSUMs; the oracle's own Gaussian F1; the oracle threshold's quantile.",
}


def oracle_threshold(scores_test, y, obs, tolerance: int = 20, grid=QUANTILE_GRID) -> tuple[float, float, float]:
    """(threshold, its quantile of the finite test scores, event F1) maximising event F1 over ``grid``.

    Uses the test labels: an upper bound on any static threshold, never a deployable rule.
    """
    s = np.asarray(scores_test, dtype=float)
    fin = s[np.isfinite(s)]
    best = (np.inf, float("nan"), -1.0)
    for q in grid:
        thr = float(np.quantile(fin, q))
        f1 = _metrics(y, obs, (s > thr).astype(int), tolerance)["event_f1"]
        if f1 > best[2]:
            best = (thr, float(q), float(f1))
    return best


def run_slice8_experiment(seed: int, scenario: str, n: int = 6000, windows=(0.3, 0.2, 0.5), rate: float = 0.005,
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
        if det not in DETECTORS8:
            continue
        cal_s = s[cal][keep]
        thr_q = false_alarm_threshold(cal_s[np.isfinite(cal_s)], rate)
        static = _metrics(y[test], obs[test], (s[test] > thr_q).astype(int), tolerance)
        o_thr, o_q, o_f1 = oracle_threshold(s[test], y[test], obs[test], tolerance)
        rows.append({"seed": seed, "scenario": scenario, "detector": det, "static_threshold": float(thr_q),
                     "static_f1": static["event_f1"], "oracle_threshold": o_thr, "oracle_quantile": o_q,
                     "oracle_f1": o_f1})
    return pd.DataFrame(rows)


def run_slice8_grid(seeds: Iterable[int] = EVAL_SEEDS, scenarios: Iterable[str] = tuple(SCENARIOS6)) -> pd.DataFrame:
    return pd.concat([run_slice8_experiment(s, sc) for sc in scenarios for s in seeds], ignore_index=True)


def _boot_ci(d: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    b = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def decide(df: pd.DataFrame) -> dict:
    rz = df[df.detector == "robust_z"].set_index(["scenario", "seed"])
    gauss_static = float(rz.loc["weekend"].static_f1.mean())
    t3 = rz.loc["weekend_t3"]
    oracle_t3 = float(t3.oracle_f1.mean())
    gain = (t3.oracle_f1 - t3.static_f1).to_numpy()
    lo, hi = _boot_ci(gain)
    return {"H1": bool(oracle_t3 >= gauss_static - 0.10), "gauss_static_f1": gauss_static, "t3_oracle_f1": oracle_t3,
            "t3_static_f1": float(t3.static_f1.mean()), "t3_gain": float(gain.mean()), "t3_gain_ci": [lo, hi],
            "H2_exhausted": bool(hi < 0.05)}


def posthoc_oracle_far(df: pd.DataFrame, n: int = 6000, windows=(0.3, 0.2, 0.5), far_budget: float = 0.005,
                       tolerance: int = 20) -> pd.DataFrame:
    """POST-HOC (after the decision; decides nothing): realised false-alarm rate on normal test points
    at the static and oracle thresholds, per seed, scenario and detector, from the thresholds in ``df``;
    plus ``budget_oracle_f1``, the best event F1 over the same grid among thresholds whose realised
    false-alarm rate stays within ``far_budget`` (also label-using, also an upper bound)."""
    out = []
    for (seed, scenario), g in df.groupby(["seed", "scenario"], sort=False):
        stream = make_drifting_stream(n=n, n_events=24, warmup=600, min_gap=80, random_state=int(seed),
                                      **SCENARIOS6[scenario])
        fit, cal, test = contiguous_windows(n, windows)
        t, x, y = stream.frame["t"].to_numpy(), stream.values, stream.labels
        obs = stream.frame["observed"].to_numpy()
        resid = residuals5(x, t, fit, stream.meta["period"], "harm_dw3")
        normal = obs[test] & (y[test] == 0)
        scores = score6(resid, fit.stop - fit.start)
        for r in g.itertuples():
            full = scores[r.detector][test]
            st = full[normal]
            fin = full[np.isfinite(full)]
            best = 0.0
            for q in QUANTILE_GRID:
                thr = float(np.quantile(fin, q))
                if (st > thr).mean() <= far_budget:
                    best = max(best, _metrics(y[test], obs[test], (full > thr).astype(int), tolerance)["event_f1"])
            out.append({"seed": seed, "scenario": scenario, "detector": r.detector,
                        "static_far": float((st > r.static_threshold).mean()),
                        "oracle_far": float((st > r.oracle_threshold).mean()), "budget_oracle_f1": float(best)})
    return pd.DataFrame(out)
