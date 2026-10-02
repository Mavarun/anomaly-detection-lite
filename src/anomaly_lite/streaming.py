"""Causal (one-pass, no look-ahead) streaming anomaly scorers.

Every scorer maps a 1-D series to a non-negative score per time step where
``score[t]`` depends only on ``x[0..t]``. Tests enforce this by perturbing
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


def _as_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("empty series")
    if not np.all(np.isfinite(x)):
        raise ValueError("series must be finite")
    return x


class SeasonalProfile:
    """Per-phase median seasonal profile fitted on a reference window."""

    def __init__(self, period: int) -> None:
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period = period
        self.profile_: Optional[np.ndarray] = None

    def fit(self, values, t=None) -> "SeasonalProfile":
        x = _as_1d(values)
        t = np.arange(x.size) if t is None else np.asarray(t, dtype=int)
        phase = t % self.period
        prof = np.full(self.period, np.nan)
        for p in range(self.period):
            m = phase == p
            if m.any():
                prof[p] = np.median(x[m])
        if np.isnan(prof).any():
            raise ValueError("reference window must cover every phase at least once")
        self.profile_ = prof
        return self

    def transform(self, values, t=None) -> np.ndarray:
        if self.profile_ is None:
            raise RuntimeError("SeasonalProfile is not fitted")
        x = _as_1d(values)
        t = np.arange(x.size) if t is None else np.asarray(t, dtype=int)
        return x - self.profile_[t % self.period]


class EWMADetector:
    """Exponentially weighted mean/variance z-score with Huber-clipped updates."""

    def __init__(self, alpha: float = 0.05, clip: float = 3.0, warmup: int = 50) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha, self.clip, self.warmup = alpha, clip, warmup

    def score(self, values) -> np.ndarray:
        x = _as_1d(values)
        n, a = x.size, self.alpha
        out = np.zeros(n)
        w = min(self.warmup, n)
        m = float(np.mean(x[:w]))
        v = float(np.var(x[:w])) if w > 1 else 1.0
        v = max(v, 1e-12)
        for t in range(n):
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
                 reset_threshold: Optional[float] = None) -> None:
        if k < 0:
            raise ValueError("k must be non-negative")
        self.k, self.reference, self.reset_threshold = k, reference, reset_threshold
        self.mu_: Optional[float] = None
        self.sd_: Optional[float] = None

    def score(self, values) -> np.ndarray:
        x = _as_1d(values)
        ref = x[: min(self.reference, x.size)]
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med))) * _MAD_TO_SD
        self.mu_, self.sd_ = med, max(mad, 1e-12)
        z = (x - self.mu_) / self.sd_
        sp = sm = 0.0
        out = np.zeros(x.size)
        for t in range(x.size):
            sp = max(0.0, sp + z[t] - self.k)
            sm = max(0.0, sm - z[t] - self.k)
            out[t] = max(sp, sm)
            if self.reset_threshold is not None and out[t] > self.reset_threshold:
                sp = sm = 0.0
        return out


class RobustZDetector:
    """Median/MAD z-score over a strictly trailing window (current point excluded)."""

    def __init__(self, window: int = 200, min_periods: Optional[int] = None) -> None:
        if window < 3:
            raise ValueError("window must be >= 3")
        self.window = window
        self.min_periods = min_periods or max(3, window // 4)

    def score(self, values) -> np.ndarray:
        x = _as_1d(values)
        s = pd.Series(x)
        past = s.shift(1)  # exclude current observation
        roll = past.rolling(self.window, min_periods=self.min_periods)
        med = roll.median()
        mad = roll.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True)
        scale = (mad * _MAD_TO_SD).clip(lower=1e-9)
        z = ((s - med).abs() / scale).fillna(0.0)
        return z.to_numpy()
