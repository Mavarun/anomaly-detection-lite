"""Point-wise, point-adjusted and event-level metrics for time-series alarms.

Three views of the same binary alarm sequence:

* **Point-wise** P/R/F1 -- every time step is a sample. Penalises long
  events heavily (an alarm on 1 of 60 points scores recall 1/60).
* **Point-adjusted (PA)** P/R/F1 (Xu et al., 2018) -- if any alarm falls
  inside a true event, the whole event counts as detected before point-wise
  scoring. Widely used, and known to inflate scores: random alarms with a
  modest rate hit most long events, so PA-F1 can look strong for a random
  detector (Kim et al., AAAI 2022). Reported here *beside* stricter metrics,
  never alone.
* **Event-level** -- recall = share of true events with >= 1 alarm in
  [start, end + tolerance]; precision = share of predicted alarm *segments*
  that overlap a (tolerance-extended) true event. Also mean detection
  delay and false-alarm segments per 1,000 normal steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _binary(a, name) -> np.ndarray:
    a = np.asarray(a).astype(int).ravel()
    if not np.all(np.isin(a, (0, 1))):
        raise ValueError(f"{name} must be binary 0/1")
    return a


def _check(y, pred):
    y, pred = _binary(y, "y_true"), _binary(pred, "y_pred")
    if y.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return y, pred


def segments(binary) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of runs of ones."""
    b = _binary(binary, "binary")
    if b.size == 0:
        return []
    d = np.diff(np.r_[0, b, 0])
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return float(p), float(r), float(f)


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}precision": self.precision, f"{prefix}recall": self.recall,
                f"{prefix}f1": self.f1}


def point_metrics(y_true, y_pred) -> PRF:
    y, p = _check(y_true, y_pred)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    return PRF(*_prf(tp, fp, fn))


def point_adjust(y_true, y_pred) -> np.ndarray:
    """Return predictions with every partially-hit true event filled in."""
    y, p = _check(y_true, y_pred)
    out = p.copy()
    for s, e in segments(y):
        if p[s : e + 1].any():
            out[s : e + 1] = 1
    return out


def point_adjusted_metrics(y_true, y_pred) -> PRF:
    return point_metrics(y_true, point_adjust(y_true, y_pred))


@dataclass(frozen=True)
class EventReport:
    precision: float
    recall: float
    f1: float
    n_true_events: int
    n_detected_events: int
    n_pred_segments: int
    n_false_segments: int
    mean_delay: float  # steps from event start to first alarm, detected events only
    false_alarms_per_1k: float  # false segments per 1000 label-0 steps

    def as_dict(self, prefix: str = "event_") -> dict:
        return {f"{prefix}{k}": v for k, v in self.__dict__.items()}


def event_metrics(y_true, y_pred, tolerance: int = 0) -> EventReport:
    """Event-level precision/recall/F1 with an optional post-event tolerance.

    ``tolerance`` extends each true event's *end* (never its start): alarms
    shortly after an event -- typical for CUSUM/EWMA lag -- count as hits
    rather than false alarms. Alarms before the event start never count.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    y, p = _check(y_true, y_pred)
    n = y.size
    true_segs = segments(y)
    windows = [(s, min(n - 1, e + tolerance)) for s, e in true_segs]
    hit_mask = np.zeros(n, dtype=bool)
    for s, e in windows:
        hit_mask[s : e + 1] = True

    detected, delays = 0, []
    for s, e in windows:
        idx = np.flatnonzero(p[s : e + 1])
        if idx.size:
            detected += 1
            delays.append(int(idx[0]))
    pred_segs = segments(p)
    false_segs = sum(1 for s, e in pred_segs if not hit_mask[s : e + 1].any())
    prec = (len(pred_segs) - false_segs) / len(pred_segs) if pred_segs else 0.0
    rec = detected / len(true_segs) if true_segs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    n_normal = int(np.sum(~hit_mask))
    return EventReport(
        precision=float(prec), recall=float(rec), f1=float(f1),
        n_true_events=len(true_segs), n_detected_events=detected,
        n_pred_segments=len(pred_segs), n_false_segments=int(false_segs),
        mean_delay=float(np.mean(delays)) if delays else float("nan"),
        false_alarms_per_1k=1000.0 * false_segs / n_normal if n_normal else float("nan"),
    )


def all_alarm_metrics(y_true, y_pred, tolerance: int = 0) -> dict:
    """Point, point-adjusted and event metrics in one flat dict."""
    out = {}
    out.update(point_metrics(y_true, y_pred).as_dict("point_"))
    out.update(point_adjusted_metrics(y_true, y_pred).as_dict("pa_"))
    out.update(event_metrics(y_true, y_pred, tolerance).as_dict("event_"))
    return out


@dataclass(frozen=True)
class AlarmCost:
    """Operational cost of an alarm sequence (slice 3).

    ``total = c_false * false alarm segments + c_miss * missed events
    + c_delay * sum of detection delays (steps) + c_busy * alarmed normal
    steps``, also given per 1,000 steps. Defaults price a page-out at 1, a
    missed incident at 10, each step of detection delay at 0.1 and each step an
    alarm stays on outside any (tolerance-extended) event at 0.02: a missed
    event costs as much as ten false pages, catching it 100 steps late costs as
    much as missing it, and an alarm stuck on for 50 normal steps costs one
    page. Without the busy term a detector that alarms *everywhere* forms one
    segment that touches every event and looks almost free (slice-3 CUSUM
    under drift: point FAR 0.97 at a cost of 1.3 per 1k). The weights are an
    assumption, so harnesses should report the components too.
    """

    total: float
    per_1k: float
    n_false: int
    n_missed: int
    delay_steps: int
    busy_steps: int = 0

    def as_dict(self, prefix: str = "cost_") -> dict:
        return {f"{prefix}{k}": v for k, v in self.__dict__.items()}


def alarm_cost(y_true, y_pred, tolerance: int = 0, c_false: float = 1.0, c_miss: float = 10.0,
               c_delay: float = 0.1, c_busy: float = 0.02) -> AlarmCost:
    """Cost of ``y_pred`` with the same hit / false-segment rules as :func:`event_metrics`."""
    if min(c_false, c_miss, c_delay, c_busy) < 0:
        raise ValueError("costs must be non-negative")
    y, p = _check(y_true, y_pred)
    n = y.size
    windows = [(s, min(n - 1, e + tolerance)) for s, e in segments(y)]
    hit_mask = np.zeros(n, dtype=bool)
    for s, e in windows:
        hit_mask[s : e + 1] = True
    missed, delay = 0, 0
    for s, e in windows:
        idx = np.flatnonzero(p[s : e + 1])
        if idx.size:
            delay += int(idx[0])
        else:
            missed += 1
    n_false = sum(1 for s, e in segments(p) if not hit_mask[s : e + 1].any())
    busy = int(np.sum((p == 1) & ~hit_mask))
    total = c_false * n_false + c_miss * missed + c_delay * delay + c_busy * busy
    return AlarmCost(total=float(total), per_1k=1000.0 * total / n if n else float("nan"),
                     n_false=int(n_false), n_missed=int(missed), delay_steps=int(delay), busy_steps=busy)
