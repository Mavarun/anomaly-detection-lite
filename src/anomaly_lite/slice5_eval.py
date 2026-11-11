"""Slice 5: a non-sinusoidal weekly shape, per-period harmonics and a prewhitened CUSUM.

Reuses the slice-4 stream, windows (fit 30% / cal 20% / test 50%), metrics, alarm cost,
threshold protocols and paired statistics (:mod:`anomaly_lite.slice4_eval`).

* Scenarios (``SCENARIOS5``): ``weekly`` (slice 4: drift plus a 672-step sinusoid of
  amplitude 1.5), ``weekend`` (the same drift plus the zero-mean weekend-dip shape of the same
  peak-to-peak range, :func:`anomaly_lite.stream_data.weekend_shape`) and ``weekend_missing``
  (plus 5% missing in gaps).
* De-seasonalisers: ``harm_dw`` (slice 4, one harmonic per period) and ``harm_dw3`` (one daily,
  three weekly harmonics, :class:`OnlineHarmonicProfile` with ``n_harmonics=(1, 3)``).
* Detectors: ``robust_z`` and ``ewma`` (slice 2), ``cusum`` (raw residual, k = 0.5),
  ``cusum_k15`` (raw residual, k = 1.5, the slice-4 dev-tuned fix) and ``cusum_pw``
  (:class:`PrewhitenedCUSUM`, textbook k = 0.5 on AR(1) innovations).
* Protocols: ``static_quantile`` and ``rolling_bulk`` at a 0.5% target false-alarm rate.

Nothing was tuned: k = 0.5 is the textbook allowance for a 1-sd innovation shift and three
weekly harmonics were fixed before any run. :data:`PREREGISTERED` holds the decision rules,
committed before the evaluation seeds 0-9 were run. Development seeds 100-105 were used only to
check that the harness runs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from anomaly_lite.drift_eval import _metrics, paired_effects, summarise  # noqa: F401
from anomaly_lite.event_metrics import segments
from anomaly_lite.slice4_eval import SCENARIOS4, WEEK, alarms4, residuals4
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.streaming import (
    CUSUMDetector,
    EWMADetector,
    PrewhitenedCUSUM,
    RobustZDetector,
    weekly_harmonic_profile,
)
from anomaly_lite import slice4_eval as _s4

SCENARIOS5 = {
    "weekly": dict(SCENARIOS4["weekly"]),
    "weekend": {**SCENARIOS4["weekly"], "season2_shape": "weekend"},
    "weekend_missing": {**SCENARIOS4["weekly_missing"], "season2_shape": "weekend"},
}
DESEASON5 = ("harm_dw", "harm_dw3")
DETECTORS5 = ("robust_z", "ewma", "cusum", "cusum_k15", "cusum_pw")
PROTOCOLS5 = ("static_quantile", "rolling_bulk")
EVAL_SEEDS = tuple(range(10))
DEV_SEEDS = tuple(range(100, 106))

# Bulk-tail family per detector for the rolling protocol (same rule as slice 4: |z|-type scores
# half-normal, accumulated scores exponential).
_s4.TAIL.setdefault("cusum_pw", "exponential")

PREREGISTERED = {
    "H1_weekly_harmonics": (
        "harm_dw3 becomes the recommended weekly de-seasonaliser iff, with the static threshold and the "
        "robust_z detector, mean event-F1(harm_dw3 - harm_dw) > 0 on 'weekend' with harm_dw3 better on "
        ">= 7 of 10 seeds AND mean event-F1(harm_dw3 - harm_dw) >= -0.01 on 'weekly' (sinusoid). Otherwise OFF."
    ),
    "H2_prewhitened_cusum": (
        "cusum_pw replaces cusum_k15 as the recommended CUSUM iff, with the static threshold on the "
        "recommended de-seasonaliser (decided by H1), mean event-F1(cusum_pw - cusum_k15) >= 0 on both "
        "'weekly' and 'weekend', cusum_pw better on >= 6 of 10 seeds in at least one of them, AND mean "
        "cost_per_1k(cusum_pw) <= 1.05 x cost_per_1k(cusum_k15) on both. Otherwise OFF (k 1.5 stays)."
    ),
    "reported_not_decided": (
        "weekend_missing, the rolling_bulk protocol, mean detection delay and raw cusum k 0.5 are reported "
        "for context only."
    ),
}


def residuals5(x, t, fit, period, how, forgetting=0.995):
    if how != "harm_dw3":
        return residuals4(x, t, fit, period, how, forgetting)
    out = residuals4(x, t, fit, period, "static", forgetting)
    rest = slice(fit.stop, None)
    out[rest] = weekly_harmonic_profile(period, WEEK, forgetting).fit(x[fit], t[fit]).transform(x[rest], t[rest])
    return out


def score5(resid: np.ndarray, fit_len: int) -> dict[str, np.ndarray]:
    return {
        "robust_z": RobustZDetector(window=200, missing="skip").score(resid),
        "ewma": EWMADetector(alpha=0.05, clip=3.0, warmup=min(200, fit_len), missing="skip").score(resid),
        "cusum": CUSUMDetector(k=0.5, reference=fit_len, missing="skip").score(resid),
        "cusum_k15": CUSUMDetector(k=1.5, reference=fit_len, missing="skip").score(resid),
        "cusum_pw": PrewhitenedCUSUM(k=0.5, reference=fit_len, missing="skip").score(resid),
    }


def run_slice5_experiment(seed: int = 0, scenario: str = "weekend", n: int = 6000, windows=(0.3, 0.2, 0.5),
                          rate: float = 0.005, tolerance: int = 20, n_events: int = 24,
                          deseason: Iterable[str] = DESEASON5, protocols: Iterable[str] = PROTOCOLS5,
                          detectors: Iterable[str] = DETECTORS5) -> pd.DataFrame:
    stream = make_drifting_stream(n=n, n_events=n_events, warmup=600, min_gap=80, random_state=seed,
                                  **SCENARIOS5[scenario])
    fit, cal, test = contiguous_windows(n, windows)
    t = stream.frame["t"].to_numpy()
    x, y = stream.values, stream.labels
    obs = stream.frame["observed"].to_numpy()
    y_te, obs_te = y[test], obs[test]
    detectors = tuple(detectors)
    rows = []
    for how in deseason:
        resid = residuals5(x, t, fit, stream.meta["period"], how)
        scores = {d: s for d, s in score5(resid, fit.stop - fit.start).items() if d in detectors}
        for proto in protocols:
            for det, s in scores.items():
                pred = alarms4(s, y, obs, cal, test, proto, det, rate)
                rows.append({"seed": seed, "scenario": scenario, "deseason": how, "protocol": proto,
                             "detector": det, "n_test_events": len(segments(y_te)),
                             **_metrics(y_te, obs_te, pred, tolerance)})
    return pd.DataFrame(rows)


def run_slice5_grid(seeds: Iterable[int] = EVAL_SEEDS, scenarios: Iterable[str] = tuple(SCENARIOS5),
                    **kwargs) -> pd.DataFrame:
    return pd.concat([run_slice5_experiment(seed=s, scenario=sc, **kwargs) for sc in scenarios for s in seeds],
                     ignore_index=True)


def _mean_diff(long, factor, a, b, metric, **where):
    sub = long
    for k, v in where.items():
        sub = sub[sub[k] == v]
    e = paired_effects(sub[sub[factor].isin([a, b])], factor, a, b, metric)
    assert len(e) == 1, e
    return e.iloc[0]


def decide(long: pd.DataFrame) -> dict:
    """Apply :data:`PREREGISTERED` to a static-threshold grid over the evaluation seeds."""
    st = long[long.protocol == "static_quantile"]
    we = _mean_diff(st, "deseason", "harm_dw", "harm_dw3", "event_f1", scenario="weekend", detector="robust_z")
    wk = _mean_diff(st, "deseason", "harm_dw", "harm_dw3", "event_f1", scenario="weekly", detector="robust_z")
    h1 = bool(we.diff_mean > 0 and we.n_b_better >= 7 and wk.diff_mean >= -0.01)
    rec = "harm_dw3" if h1 else "harm_dw"
    h2_parts, cost_ok, wins = [], [], []
    for sc in ("weekly", "weekend"):
        e = _mean_diff(st, "detector", "cusum_k15", "cusum_pw", "event_f1", scenario=sc, deseason=rec)
        c = st[(st.scenario == sc) & (st.deseason == rec)].groupby("detector").cost_per_1k.mean()
        h2_parts.append(e.diff_mean >= 0)
        wins.append(e.n_b_better >= 6)
        cost_ok.append(c["cusum_pw"] <= 1.05 * c["cusum_k15"])
    h2 = bool(all(h2_parts) and any(wins) and all(cost_ok))
    return {"H1": h1, "H1_weekend": we.to_dict(), "H1_weekly": wk.to_dict(), "recommended_deseason": rec,
            "H2": h2, "H2_f1_nonneg": [bool(v) for v in h2_parts], "H2_wins6": [bool(v) for v in wins],
            "H2_cost_ok": [bool(v) for v in cost_ok]}
