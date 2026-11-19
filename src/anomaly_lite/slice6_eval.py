"""Slice 6: heavy-tailed noise, and a clipped (Huber-type) prewhitened CUSUM.

Slice 5 listed as a weakness that every stream so far has Gaussian innovations. This slice
keeps the slice-5 ``weekend`` stream (drift, weekend-dip week, same events and labels) and
only swaps the innovation law for a unit-variance Student-t (:func:`make_labelled_stream`,
``noise_df``). The marginal sd, the AR(1) coefficient and the event magnitudes are unchanged,
so any loss is the tails alone.

* Scenarios (``SCENARIOS6``): ``weekend`` (Gaussian, the slice-5 control), ``weekend_t5`` and
  ``weekend_t3``.
* De-seasonaliser: ``harm_dw3`` only (shipped ON in slice 5).
* Detectors: ``robust_z`` (shipped), ``cusum_k15`` (shipped CUSUM), ``cusum_pw`` (slice 5,
  OFF) and ``cusum_pwc`` = ``PrewhitenedCUSUM(k=0.5, clip=3.0)``. The clip of 3 innovation sd
  was fixed before any run: it is well above the 1-sd innovation shift a 2-sd level step makes,
  and well below the 5-8 sd spikes, which robust z already catches.
* Protocol: the static held-out FAR quantile at 0.5%, as in slice 5's decisions.

:data:`PREREGISTERED` was committed before evaluation seeds 0-9 were run on any heavy-tailed
scenario; development seeds 100-105 only checked that the harness runs.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from anomaly_lite.drift_eval import _metrics, paired_effects
from anomaly_lite.event_metrics import segments
from anomaly_lite.slice4_eval import alarms4
from anomaly_lite.slice5_eval import SCENARIOS5, residuals5, score5
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.streaming import PrewhitenedCUSUM

SCENARIOS6 = {
    "weekend": dict(SCENARIOS5["weekend"]),
    "weekend_t5": {**SCENARIOS5["weekend"], "noise_df": 5.0},
    "weekend_t3": {**SCENARIOS5["weekend"], "noise_df": 3.0},
}
DETECTORS6 = ("robust_z", "cusum_k15", "cusum_pw", "cusum_pwc")
CLIP = 3.0
EVAL_SEEDS = tuple(range(10))
DEV_SEEDS = tuple(range(100, 106))

PREREGISTERED = {
    "H1_clipped_cusum": (
        "PrewhitenedCUSUM's recommended setting becomes clip=3.0 iff, with harm_dw3 and the static threshold, "
        "mean event-F1(cusum_pwc - cusum_pw) > 0 on 'weekend_t3' with cusum_pwc better on >= 7 of 10 seeds, "
        "AND mean event-F1(cusum_pwc - cusum_pw) >= -0.01 on Gaussian 'weekend', AND mean cost_per_1k(cusum_pwc) "
        "<= mean cost_per_1k(cusum_pw) on 'weekend_t3'. Otherwise clip stays None (OFF)."
    ),
    "H2_shipped_robust_z_guardrail": (
        "The shipped robust_z + harm_dw3 passes the heavy-tail guardrail iff its mean event F1 on 'weekend_t3' is "
        ">= mean F1 on Gaussian 'weekend' - 0.10. A failure ships nothing; it is reported as a weakness."
    ),
    "reported_not_decided": "weekend_t5, cusum_k15, mean delay and point FAR are reported for context only.",
}


def score6(resid, fit_len: int) -> dict:
    out = {k: v for k, v in score5(resid, fit_len).items() if k in ("robust_z", "cusum_k15", "cusum_pw")}
    out["cusum_pwc"] = PrewhitenedCUSUM(k=0.5, reference=fit_len, missing="skip", clip=CLIP).score(resid)
    return out


def run_slice6_experiment(seed: int = 0, scenario: str = "weekend_t3", n: int = 6000, windows=(0.3, 0.2, 0.5),
                          rate: float = 0.005, tolerance: int = 20, n_events: int = 24) -> pd.DataFrame:
    stream = make_drifting_stream(n=n, n_events=n_events, warmup=600, min_gap=80, random_state=seed,
                                  **SCENARIOS6[scenario])
    fit, cal, test = contiguous_windows(n, windows)
    t = stream.frame["t"].to_numpy()
    x, y = stream.values, stream.labels
    obs = stream.frame["observed"].to_numpy()
    resid = residuals5(x, t, fit, stream.meta["period"], "harm_dw3")
    rows = []
    for det, s in score6(resid, fit.stop - fit.start).items():
        pred = alarms4(s, y, obs, cal, test, "static_quantile", det, rate)
        rows.append({"seed": seed, "scenario": scenario, "deseason": "harm_dw3", "protocol": "static_quantile",
                     "detector": det, "n_test_events": len(segments(y[test])),
                     **_metrics(y[test], obs[test], pred, tolerance)})
    return pd.DataFrame(rows)


def run_slice6_grid(seeds: Iterable[int] = EVAL_SEEDS, scenarios: Iterable[str] = tuple(SCENARIOS6),
                    **kwargs) -> pd.DataFrame:
    return pd.concat([run_slice6_experiment(seed=s, scenario=sc, **kwargs) for sc in scenarios for s in seeds],
                     ignore_index=True)


def _diff(long, scenario, metric):
    sub = long[(long.scenario == scenario) & long.detector.isin(["cusum_pw", "cusum_pwc"])]
    e = paired_effects(sub, "detector", "cusum_pw", "cusum_pwc", metric)
    assert len(e) == 1, e
    return e.iloc[0]


def decide(long: pd.DataFrame) -> dict:
    """Apply :data:`PREREGISTERED` to a grid over the evaluation seeds."""
    t3 = _diff(long, "weekend_t3", "event_f1")
    g = _diff(long, "weekend", "event_f1")
    cost = long[long.scenario == "weekend_t3"].groupby("detector").cost_per_1k.mean()
    h1_parts = {"t3_gain": bool(t3.diff_mean > 0 and t3.n_b_better >= 7), "gauss_ok": bool(g.diff_mean >= -0.01),
                "cost_ok": bool(cost["cusum_pwc"] <= cost["cusum_pw"])}
    f1 = long[long.detector == "robust_z"].groupby("scenario").event_f1.mean()
    h2 = bool(f1["weekend_t3"] >= f1["weekend"] - 0.10)
    return {"H1": all(h1_parts.values()), "H1_parts": h1_parts, "H1_t3": t3.to_dict(), "H1_gauss": g.to_dict(),
            "H2": h2, "robust_z_f1": {k: float(v) for k, v in f1.items()},
            "recommended_clip": CLIP if all(h1_parts.values()) else None}
