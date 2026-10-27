"""Choosing alarm thresholds on a held-out window, not on the test window.

Protocols compared by :func:`evaluate_threshold_protocols`:

* ``cal_quantile`` (unsupervised) -- threshold = finite-sample upper
  quantile of scores on the calibration window's label-0 points so that a
  target fraction ``rate`` of normal points alarm. Labels are only used to
  drop known anomalies; with ``use_labels=False`` the whole window is
  assumed normal (contamination then raises the threshold).
* ``cal_best_f1`` (supervised) -- threshold maximising a chosen F1 on the
  labelled calibration window.
* ``test_oracle`` -- threshold maximising the same F1 *on the test window*.
  This is what many benchmark tables implicitly report; it is optimistic by
  construction and is included only as an upper reference.

Caveat: the quantile rule assumes calibration and test normal scores are
exchangeable. Streaming scores are autocorrelated and the series may
drift, so the realised test false-alarm rate is measured, not assumed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from anomaly_lite.event_metrics import (
    event_metrics,
    point_adjusted_metrics,
    point_metrics,
)

_METRICS = ("event_f1", "point_f1", "pa_f1")


def _f1(y, pred, metric: str, tolerance: int) -> float:
    if metric == "event_f1":
        return event_metrics(y, pred, tolerance).f1
    if metric == "point_f1":
        return point_metrics(y, pred).f1
    if metric == "pa_f1":
        return point_adjusted_metrics(y, pred).f1
    raise ValueError(f"metric must be one of {_METRICS}")


def false_alarm_threshold(scores, rate: float = 0.01, labels=None) -> float:
    """Finite-sample quantile so that ~``rate`` of normal points exceed it.

    Uses the ceil((n + 1)(1 - rate))-th smallest normal score (the split-
    conformal rank), returning +inf if n is too small for the requested rate.
    Alarms are ``score > threshold``.
    """
    if not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    s = np.asarray(scores, dtype=float).ravel()
    if labels is not None:
        lab = np.asarray(labels).astype(int).ravel()
        if lab.shape != s.shape:
            raise ValueError("labels must match scores")
        s = s[lab == 0]
    s = np.sort(s)
    n = s.size
    if n == 0:
        raise ValueError("no normal calibration scores")
    k = int(np.ceil((n + 1) * (1 - rate)))
    return float("inf") if k > n else float(s[k - 1])


def apply_threshold(scores, threshold: float) -> np.ndarray:
    return (np.asarray(scores, dtype=float).ravel() > threshold).astype(int)


def best_f1_threshold(
    scores,
    labels,
    metric: str = "event_f1",
    tolerance: int = 0,
    n_candidates: int = 200,
    min_quantile: float = 0.5,
    extra_candidates: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Grid-search the threshold that maximises ``metric`` on labelled data.

    Candidates are score quantiles in [min_quantile, 1). Ties go to the
    *higher* threshold (fewer alarms). ``extra_candidates`` are added to
    the grid (finite values only). Returns (threshold, best_f1).
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}")
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).astype(int).ravel()
    if s.shape != y.shape:
        raise ValueError("labels must match scores")
    qs = np.linspace(min_quantile, 0.9999, n_candidates)
    cands = np.quantile(s, qs)
    if extra_candidates is not None:
        extra = np.asarray(extra_candidates, dtype=float).ravel()
        cands = np.r_[cands, extra[np.isfinite(extra)]]
    cands = np.unique(cands)
    best_t, best_f = float(cands[-1]), -1.0
    for t in cands[::-1]:  # high -> low so ties keep the higher threshold
        f = _f1(y, apply_threshold(s, t), metric, tolerance)
        if f > best_f + 1e-12:
            best_t, best_f = float(t), f
    return best_t, float(best_f)


def evaluate_threshold_protocols(
    scores,
    labels,
    cal: slice,
    test: slice,
    rate: float = 0.01,
    metric: str = "event_f1",
    tolerance: int = 0,
    use_labels_for_quantile: bool = True,
) -> pd.DataFrame:
    """Score the three threshold protocols on the test window.

    Only ``scores[cal]`` / ``labels[cal]`` inform the two calibrated
    thresholds; ``test_oracle`` peeks at test labels by design.
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).astype(int).ravel()
    s_cal, y_cal, s_te, y_te = s[cal], y[cal], s[test], y[test]

    thr = {
        "cal_quantile": false_alarm_threshold(
            s_cal, rate, labels=y_cal if use_labels_for_quantile else None
        ),
        "cal_best_f1": best_f1_threshold(s_cal, y_cal, metric, tolerance)[0],
    }
    # oracle grid includes the calibrated thresholds, so it upper-bounds them
    thr["test_oracle"] = best_f1_threshold(
        s_te, y_te, metric, tolerance, extra_candidates=np.array(list(thr.values()))
    )[0]
    rows = []
    normal_te = y_te == 0
    for name, t in thr.items():
        pred = apply_threshold(s_te, t)
        ev = event_metrics(y_te, pred, tolerance)
        rows.append({
            "protocol": name,
            "threshold": t,
            "point_f1": point_metrics(y_te, pred).f1,
            "pa_f1": point_adjusted_metrics(y_te, pred).f1,
            "event_f1": ev.f1,
            "event_precision": ev.precision,
            "event_recall": ev.recall,
            "mean_delay": ev.mean_delay,
            "false_alarms_per_1k": ev.false_alarms_per_1k,
            "point_far": float(pred[normal_te].mean()) if normal_te.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def rolling_false_alarm_thresholds(
    scores,
    rate: float = 0.005,
    window: int = 1200,
    start: int = 0,
    step: int = 50,
    valid=None,
) -> np.ndarray:
    """Walk-forward, label-free threshold: re-calibrated on the trailing window only.

    ``thr[t]`` (for ``t >= start``) is :func:`false_alarm_threshold` on the
    scores in ``[t0 - window, t0)`` where ``t0`` is the latest refresh point
    ``<= t`` (refreshes every ``step`` steps from ``start``). No label is used:
    the trailing window is assumed normal, so events inside it raise the
    threshold (conservative). ``valid`` (bool mask) drops points from the
    calibration sample, e.g. missing observations whose score is a placeholder
    0. Entries before ``start`` are ``nan``. Unlike a threshold fixed on one
    calibration window, this tracks drift in the score distribution; the
    price is that a long event slowly becomes part of the reference.
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    if window < 10 or step < 1 or not 0 <= start <= n:
        raise ValueError("need window >= 10, step >= 1 and 0 <= start <= n")
    ok = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).ravel()
    if ok.shape != s.shape:
        raise ValueError("valid must match scores")
    thr = np.full(n, np.nan)
    for t0 in range(start, n, step):
        lo = max(0, t0 - window)
        ref = s[lo:t0][ok[lo:t0]]
        val = false_alarm_threshold(ref, rate) if ref.size >= 10 else float("inf")
        thr[t0 : min(n, t0 + step)] = val
    return thr


def apply_thresholds(scores, thresholds) -> np.ndarray:
    """Point-wise ``score > threshold`` (``nan`` thresholds never alarm)."""
    s = np.asarray(scores, dtype=float).ravel()
    t = np.asarray(thresholds, dtype=float).ravel()
    if s.shape != t.shape:
        raise ValueError("thresholds must match scores")
    return np.where(np.isnan(t), 0, s > t).astype(int)


def rolling_rescaled_thresholds(
    scores,
    base_threshold: float,
    ref: slice,
    window: int = 1200,
    start: int = 0,
    step: int = 50,
    valid=None,
) -> np.ndarray:
    """Walk-forward threshold that keeps the calibrated rank but tracks the score *scale*.

    ``thr[t] = base_threshold * median(trailing window) / median(scores[ref])``
    with the trailing window ``[t0 - window, t0)`` refreshed every ``step``.
    The tail quantile of a label-free trailing window is set by whatever
    events it holds (prevalence ~5% vs a 0.5% target rate), so
    :func:`rolling_false_alarm_thresholds` ends up far too high; a median moves
    with a drifting scale but barely with a few percent of events. Labels
    enter only through ``base_threshold`` (calibrated on ``ref``).
    """
    s = np.asarray(scores, dtype=float).ravel()
    n = s.size
    if window < 10 or step < 1 or not 0 <= start <= n:
        raise ValueError("need window >= 10, step >= 1 and 0 <= start <= n")
    ok = np.ones(n, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).ravel()
    if ok.shape != s.shape:
        raise ValueError("valid must match scores")
    base_ref = s[ref][ok[ref]]
    m0 = float(np.median(base_ref)) if base_ref.size else 0.0
    thr = np.full(n, np.nan)
    if not np.isfinite(base_threshold) or m0 <= 0:
        thr[start:] = base_threshold
        return thr
    for t0 in range(start, n, step):
        lo = max(0, t0 - window)
        w = s[lo:t0][ok[lo:t0]]
        m = float(np.median(w)) if w.size >= 10 else m0
        thr[t0 : min(n, t0 + step)] = base_threshold * max(m, 1e-12) / m0
    return thr
