"""End-to-end streaming evaluation: detectors x threshold protocols x metrics.

Per seed (see :func:`run_streaming_experiment`):

1. Draw a labelled stream (:func:`make_labelled_stream`).
2. Split time into consecutive windows: ``fit`` (seasonal profile, CUSUM
   reference), ``cal`` (threshold calibration), ``test`` (all reported
   numbers). Nothing from a later window informs an earlier decision.
3. Subtract a per-phase median seasonal profile fitted on ``fit`` only.
4. Score with EWMA, CUSUM, robust-z and a ``random`` control (i.i.d.
   uniform scores -- any metric that rates it well is suspect).
5. Thresholds: ``cal_quantile`` (target normal false-alarm rate),
   ``cal_best_f1`` (event-F1 on labelled cal window), ``test_oracle``
   (optimistic reference). Report point / point-adjusted / event metrics on
   ``test``, plus threshold-free point ROC-AUC and AP, plus per-event-type
   recall under ``cal_best_f1``.

:func:`nile_cusum_alarms` applies CUSUM to the public Nile annual-flow
series bundled with statsmodels (1871-1970; a level drop around 1898-1899
is the textbook change point, Cobb 1978).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from anomaly_lite.event_metrics import segments
from anomaly_lite.stream_data import contiguous_windows, make_labelled_stream
from anomaly_lite.streaming import (
    CUSUMDetector,
    EWMADetector,
    RobustZDetector,
    SeasonalProfile,
)
from anomaly_lite.thresholds import apply_threshold, evaluate_threshold_protocols


def score_stream(residual: np.ndarray, fit_len: int, seed: int) -> dict[str, np.ndarray]:
    """Scores for every detector on a de-seasonalised residual."""
    rng = np.random.default_rng(seed + 777)
    return {
        "ewma": EWMADetector(alpha=0.05, clip=3.0, warmup=min(200, fit_len)).score(residual),
        "cusum": CUSUMDetector(k=0.5, reference=fit_len).score(residual),
        "robust_z": RobustZDetector(window=200).score(residual),
        "random": rng.uniform(size=residual.size),
    }


def _type_recall(y_te, pred_te, event_ids_te, events, tolerance):
    """Recall per injected event type (event counted if any alarm in window)."""
    out = {}
    n = len(y_te)
    for kind in ("spike", "level_shift", "variance_burst"):
        ids = [e["event_id"] for e in events if e["type"] == kind]
        hit = tot = 0
        for eid in ids:
            idx = np.flatnonzero(event_ids_te == eid)
            if idx.size == 0:
                continue
            tot += 1
            lo, hi = idx[0], min(n - 1, idx[-1] + tolerance)
            hit += int(pred_te[lo : hi + 1].any())
        out[f"recall_{kind}"] = hit / tot if tot else float("nan")
    return out


def run_streaming_experiment(
    seed: int = 0,
    n: int = 6000,
    windows: tuple[float, float, float] = (0.3, 0.2, 0.5),
    rate: float = 0.005,
    tolerance: int = 20,
    n_events: int = 24,
) -> pd.DataFrame:
    """One seed -> long frame (detector x protocol) of test-window metrics."""
    stream = make_labelled_stream(n=n, n_events=n_events, warmup=600, min_gap=80,
                                  random_state=seed)
    fit, cal, test = contiguous_windows(n, windows)
    t = stream.frame["t"].to_numpy()
    x, y = stream.values, stream.labels
    prof = SeasonalProfile(period=stream.meta["period"]).fit(x[fit], t[fit])
    resid = prof.transform(x, t)
    scores = score_stream(resid, fit_len=fit.stop - fit.start, seed=seed)

    ev_ids = stream.frame["event_id"].to_numpy()
    y_te = y[test]
    rows = []
    for det, s in scores.items():
        prot = evaluate_threshold_protocols(s, y, cal, test, rate=rate, metric="event_f1",
                                            tolerance=tolerance)
        auc = float(roc_auc_score(y_te, s[test]))
        ap = float(average_precision_score(y_te, s[test]))
        for _, r in prot.iterrows():
            row = {"seed": seed, "detector": det, **r.to_dict(),
                   "point_roc_auc": auc, "point_ap": ap,
                   "n_test_events": len(segments(y_te))}
            pred_te = apply_threshold(s[test], r["threshold"])
            row.update(_type_recall(y_te, pred_te, ev_ids[test], stream.events, tolerance))
            rows.append(row)
    return pd.DataFrame(rows)


def run_streaming_grid(seeds: Iterable[int] = range(10), **kwargs) -> pd.DataFrame:
    return pd.concat([run_streaming_experiment(seed=s, **kwargs) for s in seeds],
                     ignore_index=True)


def summarise_streaming(long: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    agg = long.groupby(["detector", "protocol"], sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def load_nile() -> pd.DataFrame:
    """Public Nile annual flow (10^8 m^3) at Aswan, 1871-1970, via statsmodels."""
    import statsmodels.api as sm

    df = sm.datasets.nile.load_pandas().data.copy()
    df["year"] = df["year"].astype(int)
    return df[["year", "volume"]]


def nile_cusum_alarms(
    references: Iterable[int] = (15, 20, 25),
    thresholds: Iterable[float] = (4.0, 5.0),
    k: float = 0.5,
) -> pd.DataFrame:
    """First CUSUM alarm year for each (reference length, threshold h)."""
    df = load_nile()
    x, yr = df["volume"].to_numpy(dtype=float), df["year"].to_numpy()
    rows = []
    for ref in references:
        det = CUSUMDetector(k=k, reference=ref)
        s = det.score(x)
        for h in thresholds:
            above = np.flatnonzero(s > h)
            rows.append({"reference_years": ref, "reference_end": int(yr[ref - 1]), "h": h,
                         "ref_median": det.mu_, "ref_robust_sd": det.sd_,
                         "first_alarm_year": int(yr[above[0]]) if above.size else None})
    return pd.DataFrame(rows)
