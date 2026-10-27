"""Causal (one-pass, no look-ahead) streaming anomaly scorers.

Every scorer maps a 1-D series to a non-negative score per time step where
``score[t]`` depends only on ``x[0..t]`` -- with one exception: CUSUM
standardises with the median/MAD of its first ``reference`` points, so its
scores *inside* that reference window are in-sample (Phase I) and only
scores at ``t >= reference`` are causal. Harnesses keep the reference inside
the fit window and score only later windows. Tests enforce this by perturbing
the future and checking that past scores do not move. Thresholds are *not*
chosen here -- see :mod:`anomaly_lite.thresholds` -- so detectors can be
compared on threshold-free ranking metrics as well as on alarms.

* :class:`SeasonalProfile` -- per-phase median profile fitted on a
  reference window; subtracting it removes a fixed seasonal pattern
  without using future data.
* :class:`EWMADetector` -- |x_t - m_{t-1}| / s_{t-1} with exponentially
  weighted mean/variance. Updates use a Huber-clipped residual so a single
  spike does not inflate the variance for hundreds of steps. Slowly
  absorbs persistent level shifts (by design of the EWMA).
* :class:`CUSUMDetector` -- two-sided Page CUSUM on standardised
  residuals, S+_t = max(0, S+_{t-1} + z_t - k), S-_t likewise; score =
  max(S+, S-). Accumulates small persistent shifts but keeps scoring high
  for ~S/k steps after an event ends (lagged recovery).
* :class:`RobustZDetector` -- |x_t - median(W)| / (1.4826 MAD(W)) over a
  strictly trailing window W = x[t-w..t-1]. Insensitive to outliers inside
  W, but a long event eventually becomes the window's own baseline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

_MAD_TO_SD = 1.4826


MISSING = ("raise", "skip")


def _as_1d(x, missing: str = "raise") -> np.ndarray:
    """1-D float array; NaN marks a missing point only when ``missing="skip"``.

    ``+-inf`` is always rejected. With ``"skip"`` a detector scores a missing
    point 0 (no alarm is possible on data that never arrived) and does not
    update its state from it, so a gap neither inflates nor resets the scale.
    """
    if missing not in MISSING:
        raise ValueError(f"missing must be one of {MISSING}")
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("empty series")
    if np.isinf(x).any() or (missing == "raise" and np.isnan(x).any()):
        raise ValueError("series must be finite (pass missing='skip' to allow NaN gaps)")
    if np.isnan(x).all():
        raise ValueError("series has no observed point")
    return x


class SeasonalProfile:
    """Per-phase median seasonal profile fitted on a reference window."""

    def __init__(self, period: int, missing: str = "raise") -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period = period
        self.missing = missing
        self.profile_: Optional[np.ndarray] = None

    def fit(self, values, t=None) -> "SeasonalProfile":
        x = _as_1d(values, self.missing)
        t = np.arange(x.size) if t is None else np.asarray(t, dtype=int)
        phase = t % self.period
        prof = np.full(self.period, np.nan)
        for p in range(self.period):
            m = (phase == p) & ~np.isnan(x)
            if m.any():
                prof[p] = np.median(x[m])
        if np.isnan(prof).any():
            raise ValueError("reference window must cover every phase at least once")
        self.profile_ = prof
        return self

    def transform(self, values, t=None) -> np.ndarray:
        if self.profile_ is None:
            raise RuntimeError("SeasonalProfile is not fitted")
        x = _as_1d(values, self.missing)
        t = np.arange(x.size) if t is None else np.asarray(t, dtype=int)
        return x - self.profile_[t % self.period]


class EWMADetector:
    """Exponentially weighted mean/variance z-score with Huber-clipped updates."""

    def __init__(self, alpha: float = 0.05, clip: float = 3.0, warmup: int = 50,
                 missing: str = "raise") -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha, self.clip, self.warmup, self.missing = alpha, clip, warmup, missing

    def score(self, values) -> np.ndarray:
        x = _as_1d(values, self.missing)
        n, a = x.size, self.alpha
        out = np.zeros(n)
        w = min(self.warmup, n)
        head = x[:w][~np.isnan(x[:w])]
        if head.size == 0:
            head = x[~np.isnan(x)][:1]
        m = float(np.mean(head))
        v = float(np.var(head)) if head.size > 1 else 1.0
        v = max(v, 1e-12)
        for t in range(n):
            if np.isnan(x[t]):
                continue
            s = np.sqrt(v)
            r = x[t] - m
            if t >= w:
                out[t] = abs(r) / s
            r_c = float(np.clip(r, -self.clip * s, self.clip * s))
            m = m + a * r_c
            v = max((1 - a) * v + a * r_c * r_c, 1e-12)
        return out


class CUSUMDetector:
    """Two-sided Page CUSUM on residuals standardised by a reference window.

    ``k`` is the allowance in standard-deviation units (k = delta/2 is the
    classical choice for detecting a mean shift of size delta). If
    ``reset_threshold`` is given, both sums reset to zero after crossing it
    (alarm-and-restart scheme); by default they do not reset so the score
    can be thresholded afterwards.
    """

    def __init__(self, k: float = 0.5, reference: int = 500,
                 reset_threshold: Optional[float] = None, missing: str = "raise") -> None:
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k, self.reference, self.reset_threshold = k, reference, reset_threshold
        self.missing = missing
        self.mu_: Optional[float] = None
        self.sd_: Optional[float] = None

    def score(self, values) -> np.ndarray:
        x = _as_1d(values, self.missing)
        ref = x[: min(self.reference, x.size)]
        ref = ref[~np.isnan(ref)]
        if ref.size == 0:
            raise ValueError("CUSUM reference window has no observed point")
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med))) * _MAD_TO_SD
        self.mu_, self.sd_ = med, max(mad, 1e-12)
        z = (x - self.mu_) / self.sd_
        sp = sm = 0.0
        out = np.zeros(x.size)
        for t in range(x.size):
            if np.isnan(z[t]):  # missing: hold the sums, no score
                continue
            sp = max(0.0, sp + z[t] - self.k)
            sm = max(0.0, sm - z[t] - self.k)
            out[t] = max(sp, sm)
            if self.reset_threshold is not None and out[t] > self.reset_threshold:
                sp = sm = 0.0
        return out


class RobustZDetector:
    """Median/MAD z-score over a strictly trailing window (current point excluded)."""

    def __init__(self, window: int = 200, min_periods: Optional[int] = None,
                 missing: str = "raise") -> None:
        if window < 3:
            raise ValueError("window must be >= 3")
        self.window = window
        self.min_periods = min_periods or max(3, window // 4)
        self.missing = missing

    def score(self, values) -> np.ndarray:
        # The trailing window spans ``window`` time steps; missing points inside
        # it are ignored (pandas rolling skips NaN), and a missing point scores 0.
        x = _as_1d(values, self.missing)
        s = pd.Series(x)
        past = s.shift(1)  # exclude current observation
        roll = past.rolling(self.window, min_periods=self.min_periods)
        med = roll.median()
        mad = roll.apply(lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True)
        scale = (mad * _MAD_TO_SD).clip(lower=1e-9)
        z = ((s - med).abs() / scale).fillna(0.0)
        return z.to_numpy()


class RollingVarianceDetector:
    """Explicit variance detector: short-window vs long-window robust scale of first differences.

    score_t = max(0, log(S_t / L_t)) where S_t is the MAD-scale of the last
    ``short`` first differences (ending at t, so causal) and L_t the MAD-scale
    of the ``long`` differences *before* that short window. Differencing
    removes the level, so level shifts barely move the score (only the one
    step where the shift happens) while a variance burst raises S_t for its
    whole length. Added in slice 3 because no slice-2 detector modelled
    variance; a mean-zero burst was caught only through its large points.
    """

    def __init__(self, short: int = 20, long: int = 400, min_long: Optional[int] = None,
                 missing: str = "raise") -> None:
        if short < 5 or long < 2 * short:
            raise ValueError("need short >= 5 and long >= 2 * short")
        self.short, self.long = short, long
        self.min_long = min_long or long // 4
        self.missing = missing

    def score(self, values) -> np.ndarray:
        x = _as_1d(values, self.missing)
        d = pd.Series(np.r_[np.nan, np.diff(x)])  # NaN at gaps and at t = 0
        absdev = lambda w: np.nanmedian(np.abs(w - np.nanmedian(w)))  # noqa: E731
        s = d.rolling(self.short, min_periods=max(5, self.short // 2)).apply(absdev, raw=True)
        lg = d.shift(self.short).rolling(self.long, min_periods=self.min_long).apply(absdev, raw=True)
        ratio = (s.clip(lower=1e-12) / lg.clip(lower=1e-12)).to_numpy()
        out = np.log(ratio)
        out = np.where(np.isfinite(out), np.maximum(out, 0.0), 0.0)
        out[np.isnan(x)] = 0.0
        return out
