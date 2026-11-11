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


class OnlineSeasonalProfile:
    """Causal additive Holt-Winters de-seasonaliser with Huber-clipped updates (slice 3).

    ``SeasonalProfile`` is fitted once on an early window, so a drifting level
    or a growing seasonal amplitude leaks into every later residual. Here the
    level ``l`` and the per-phase offsets ``s[p]`` keep adapting:

        r_t = x_t - (l_{t-1} + s_{t-1}[t mod P])       (residual: past state only)
        e_t = clip(r_t, -c * sigma, c * sigma)          (an anomaly barely moves it)
        l_t = l_{t-1} + alpha * e_t
        s_t[t mod P] = s_{t-1}[t mod P] + gamma * (1 - alpha) * e_t

    ``sigma`` is the MAD-scale of the fit-window residual of a static profile.
    ``fit`` initialises ``l`` / ``s`` from that window (static per-phase
    medians), and ``transform`` runs from the first point of its input, which
    must start right after the fit window (``t`` gives the phase). Missing
    points (``missing="skip"``) get a NaN residual and no update. The price of
    adapting: a long level shift is partly absorbed into ``l``, at a rate
    capped by the clip. Each phase is updated once per period, so ``gamma``
    = 0.1 remembers about ten cycles (0.05 lagged a doubling amplitude: late
    residual sd 0.53 vs 0.39 on a 0.3-noise toy series, see the tests).
    """

    def __init__(self, period: int, alpha: float = 0.005, gamma: float = 0.1, clip: float = 2.0,
                 missing: str = "raise") -> None:
        if period < 2 or not 0 < alpha < 1 or not 0 <= gamma < 1 or clip <= 0:
            raise ValueError("need period >= 2, 0 < alpha < 1, 0 <= gamma < 1, clip > 0")
        self.period, self.alpha, self.gamma, self.clip, self.missing = period, alpha, gamma, clip, missing
        self.level_: Optional[float] = None
        self.season_: Optional[np.ndarray] = None
        self.sigma_: Optional[float] = None

    def fit(self, values, t=None) -> "OnlineSeasonalProfile":
        x = _as_1d(values, self.missing)
        t = np.arange(x.size) if t is None else np.asarray(t, dtype=int)
        static = SeasonalProfile(self.period, missing=self.missing).fit(x, t)
        lvl = float(np.nanmedian(static.profile_))
        self.level_, self.season_ = lvl, static.profile_ - lvl
        r = static.transform(x, t)
        r = r[~np.isnan(r)]
        self.sigma_ = max(float(np.median(np.abs(r - np.median(r)))) * _MAD_TO_SD, 1e-9)
        return self

    def transform(self, values, t=None) -> np.ndarray:
        if self.level_ is None:
            raise RuntimeError("OnlineSeasonalProfile is not fitted")
        x = _as_1d(values, self.missing)
        t = np.arange(x.size) if t is None else np.asarray(t, dtype=int)
        lvl, seas = float(self.level_), self.season_.copy()
        a, g, c = self.alpha, self.gamma, self.clip * float(self.sigma_)
        out = np.full(x.size, np.nan)
        for i in range(x.size):
            if np.isnan(x[i]):
                continue
            p = int(t[i]) % self.period
            r = x[i] - (lvl + seas[p])
            out[i] = r
            e = min(max(r, -c), c)
            lvl += a * e
            seas[p] += g * (1.0 - a) * e
        return out


class OnlineHarmonicProfile:
    """Causal harmonic de-seasonaliser: level + sin/cos terms per period, by robust RLS (slice 4).

    ``OnlineSeasonalProfile`` keeps one offset per phase of a *single* period.
    A second cycle (a weekly rhythm over a daily one) is invisible to it and
    leaks into the level, and a period of 672 would need ~2 observations per
    phase in a 1,800-step fit window. Here the mean is a small linear model

        m_t = b0 + sum_p sum_{h=1..H} a_ph sin(2 pi h t / P_p) + c_ph cos(2 pi h t / P_p)

    with 1 + 2 H n_periods coefficients, so a long period costs only two
    parameters. ``fit`` solves it on the fit window by a few Huber-reweighted
    least-squares passes (anomalies in the window barely move it).
    ``transform`` then runs recursive least squares with forgetting factor
    ``forgetting`` (memory ~ 1 / (1 - forgetting) steps) from the first point of
    its input:

        r_t = x_t - phi_t' theta_{t-1}                 (residual: past state only)
        e_t = clip(r_t, -c sigma, c sigma)             (an anomaly barely moves theta)
        K_t = P phi / (lambda + phi' P phi);  theta += K_t e_t;  P = (P - K_t phi' P) / lambda

    so the level, the amplitudes and the phases all drift slowly with the data.
    Missing points (``missing="skip"``) get a NaN residual and no update.
    ``t`` must be the absolute time index. Costs: a coarse sinusoid can not
    follow a non-sinusoidal pattern with ``n_harmonics`` = 1, and a long event
    is partly absorbed into the level at a rate capped by the clip.

    Slice 5: ``n_harmonics`` may be a tuple with one count per period, e.g.
    ``periods=(96, 672), n_harmonics=(1, 3)`` keeps the daily sinusoid and gives
    the weekly cycle three harmonics (a non-sinusoidal weekend dip) for four more
    parameters.
    """

    def __init__(self, periods=(96,), n_harmonics: int = 1, forgetting: float = 0.995,
                 clip: float = 2.0, missing: str = "raise") -> None:
        periods = tuple(int(p) for p in np.atleast_1d(periods))
        harm = tuple(int(h) for h in np.atleast_1d(n_harmonics))
        if len(harm) == 1:
            harm = harm * len(periods)
        if not periods or min(periods) < 2 or len(harm) != len(periods) or min(harm) < 1:
            raise ValueError("need periods >= 2 and n_harmonics >= 1 (one int, or one per period)")
        if not 0.9 < forgetting <= 1.0 or clip <= 0:
            raise ValueError("need 0.9 < forgetting <= 1 and clip > 0")
        self.periods = periods
        self.harmonics = harm  # per period (slice 5); n_harmonics kept for the old single-int API
        self.n_harmonics = harm[0] if len(set(harm)) == 1 else harm
        self.forgetting, self.clip, self.missing = forgetting, clip, missing
        self.theta_: Optional[np.ndarray] = None
        self.P_: Optional[np.ndarray] = None
        self.sigma_: Optional[float] = None

    def _design(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        cols = [np.ones_like(t)]
        for p, nh in zip(self.periods, self.harmonics):
            for h in range(1, nh + 1):
                w = 2.0 * np.pi * h * t / p
                cols += [np.sin(w), np.cos(w)]
        return np.column_stack(cols)

    def fit(self, values, t=None) -> "OnlineHarmonicProfile":
        x = _as_1d(values, self.missing)
        t = np.arange(x.size) if t is None else np.asarray(t, dtype=int)
        ok = ~np.isnan(x)
        phi, y = self._design(t[ok]), x[ok]
        if y.size < 4 * phi.shape[1]:
            raise ValueError("fit window too short for this many harmonic terms")
        w = np.ones(y.size)
        for _ in range(5):  # Huber IRLS
            sw = np.sqrt(w)
            theta = np.linalg.lstsq(phi * sw[:, None], y * sw, rcond=None)[0]
            r = y - phi @ theta
            s = max(float(np.median(np.abs(r - np.median(r)))) * _MAD_TO_SD, 1e-9)
            u = np.abs(r) / (1.5 * s)
            w = np.where(u <= 1.0, 1.0, 1.0 / u)
        self.theta_, self.sigma_ = theta, s
        n_eff = 1.0 / (1.0 - self.forgetting) if self.forgetting < 1 else float(y.size)
        gram = phi.T @ phi * min(1.0, n_eff / y.size)
        self.P_ = np.linalg.inv(gram + 1e-9 * np.eye(gram.shape[0]))
        return self

    def transform(self, values, t=None) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("OnlineHarmonicProfile is not fitted")
        x = _as_1d(values, self.missing)
        t = np.arange(x.size) if t is None else np.asarray(t, dtype=int)
        phi = self._design(t)
        th, P, lam = self.theta_.copy(), self.P_.copy(), self.forgetting
        c = self.clip * float(self.sigma_)
        out = np.full(x.size, np.nan)
        for i in range(x.size):
            if np.isnan(x[i]):
                continue
            f = phi[i]
            r = x[i] - f @ th
            out[i] = r
            e = min(max(r, -c), c)
            Pf = P @ f
            k = Pf / (lam + f @ Pf)
            th = th + k * e
            P = (P - np.outer(k, Pf)) / lam
            P = 0.5 * (P + P.T)
        return out


class FrozenReferenceCUSUM:
    """Two-sided Page CUSUM with a *self-updating but freezable* reference (slice 4).

    ``CUSUMDetector`` standardises with a fixed fit-window median / MAD, so a
    slow drift pushes the sums up for ever (87% of normal test points alarmed in
    slice 3). Re-estimating the reference online cures that, but a reference
    that keeps following the data also absorbs the very shift the CUSUM is
    trying to accumulate (slice 3: F1 0.139 with an online level). The usual
    compromise is used here: the reference (EW mean ``m``, EW scale ``s``)
    adapts only while the detector looks calm, and is frozen while either sum is
    above ``freeze_at``:

        z_t = (x_t - m) / s                 (past reference only)
        S+_t = max(0, S+_{t-1} + z_t - k),  S-_t likewise,  score_t = max(S+, S-)
        if score_{t-1} < freeze_at:         # calm: let the reference follow drift
            e = clip(x_t - m, +-c s);  m += alpha e;  s^2 += beta (e^2 - s^2)

    A permanent change would freeze the reference for ever, so after
    ``max_freeze`` consecutive frozen steps both sums reset and the reference
    re-anchors on the last ``anchor`` points (the new normal). Input should
    already be de-seasonalised. ``reference`` points initialise ``m`` / ``s`` by
    median / MAD (in-sample, like :class:`CUSUMDetector`; score only later
    windows). A missing point (``missing="skip"``) holds all state and scores 0.
    """

    def __init__(self, k: float = 0.5, reference: int = 500, alpha: float = 0.005, beta: float = 0.01,
                 clip: float = 3.0, freeze_at: float = 2.0, max_freeze: int = 150, anchor: int = 30,
                 missing: str = "raise") -> None:
        if k < 0 or not 0 < alpha < 1 or not 0 < beta < 1 or clip <= 0 or freeze_at <= 0:
            raise ValueError("need k >= 0, alpha/beta in (0, 1), clip > 0, freeze_at > 0")
        if max_freeze < 1 or anchor < 5:
            raise ValueError("need max_freeze >= 1 and anchor >= 5")
        self.k, self.reference, self.alpha, self.beta, self.clip = k, reference, alpha, beta, clip
        self.freeze_at, self.max_freeze, self.anchor, self.missing = freeze_at, max_freeze, anchor, missing
        self.n_reanchor_ = 0

    def score(self, values) -> np.ndarray:
        x = _as_1d(values, self.missing)
        ref = x[: min(self.reference, x.size)]
        ref = ref[~np.isnan(ref)]
        if ref.size == 0:
            raise ValueError("reference window has no observed point")
        m = float(np.median(ref))
        s = max(float(np.median(np.abs(ref - m))) * _MAD_TO_SD, 1e-9)
        s0 = s
        sp = sm = 0.0
        frozen = 0
        recent: list[float] = []
        out = np.zeros(x.size)
        self.n_reanchor_ = 0
        for t in range(x.size):
            if np.isnan(x[t]):
                continue
            z = (x[t] - m) / s
            calm = max(sp, sm) < self.freeze_at
            sp = max(0.0, sp + z - self.k)
            sm = max(0.0, sm - z - self.k)
            out[t] = max(sp, sm)
            recent.append(x[t])
            if len(recent) > self.anchor:
                recent.pop(0)
            if calm:
                frozen = 0
                e = min(max(x[t] - m, -self.clip * s), self.clip * s)
                m += self.alpha * e
                s = max(np.sqrt(s * s + self.beta * (e * e - s * s)), 0.25 * s0)
            else:
                frozen += 1
                if frozen >= self.max_freeze and len(recent) >= self.anchor:
                    m = float(np.median(recent))
                    sp = sm = 0.0
                    frozen = 0
                    self.n_reanchor_ += 1
        return out


class PrewhitenedCUSUM:
    """Two-sided Page CUSUM on AR(1) innovations of the residual (slice 5).

    The textbook allowance k = 0.5 assumes i.i.d. standardised input. On AR(1)
    noise with coefficient phi the partial sums have variance inflated by about
    (1 + phi) / (1 - phi) (4x at phi = 0.6), so a k = 0.5 CUSUM on raw residuals
    alarms constantly; slice 4 papered over this by tuning k to 1.5 on dev seeds.
    Here the dependence is removed instead. On the reference window (the first
    ``reference`` points) the residual is standardised by median / MAD, phi is the
    lag-1 autocorrelation of the standardised reference (clipped to +-0.95), and
    the innovation scale is the MAD of ``z_t - phi z_{t-1}`` there. Then

        e_t = (z_t - phi^g z_{t-g}) / sd_e      (g = steps since the last observed point)

    feeds an ordinary two-sided CUSUM with allowance ``k`` (in innovation sd).
    A level shift of delta (in residual sd) becomes an innovation mean shift of
    delta (1 - phi) / sqrt(1 - phi^2): for phi = 0.6 a 2-sd step is a 1-sd
    innovation step, for which k = 0.5 is the classical choice. Cost: the
    sustained signal is smaller than the raw one by (1 - phi), so slow or short
    shifts are found later, and a single spike also produces a negative echo
    -phi spike on the next step.
    """

    def __init__(self, k: float = 0.5, reference: int = 500, max_phi: float = 0.95,
                 missing: str = "raise") -> None:
        if k < 0 or not 0 <= max_phi < 1:
            raise ValueError("need k >= 0 and 0 <= max_phi < 1")
        self.k, self.reference, self.max_phi, self.missing = k, reference, max_phi, missing
        self.mu_: Optional[float] = None
        self.sd_: Optional[float] = None
        self.phi_: Optional[float] = None
        self.sd_e_: Optional[float] = None

    def innovations(self, values) -> np.ndarray:
        x = _as_1d(values, self.missing)
        ref = x[: min(self.reference, x.size)]
        ok = ~np.isnan(ref)
        if ok.sum() < 10:
            raise ValueError("prewhitened CUSUM reference window needs >= 10 observed points")
        med = float(np.median(ref[ok]))
        self.mu_ = med
        self.sd_ = max(float(np.median(np.abs(ref[ok] - med))) * _MAD_TO_SD, 1e-12)
        zr = (ref - med) / self.sd_
        pair = ~np.isnan(zr[1:]) & ~np.isnan(zr[:-1])
        a, b = zr[1:][pair], zr[:-1][pair]
        phi = float(np.sum(a * b) / max(np.sum(b * b), 1e-12)) if pair.sum() >= 5 else 0.0
        self.phi_ = float(np.clip(phi, -self.max_phi, self.max_phi))
        er = a - self.phi_ * b
        self.sd_e_ = max(float(np.median(np.abs(er - np.median(er)))) * _MAD_TO_SD, 1e-12)
        z = (x - med) / self.sd_
        out = np.full(x.size, np.nan)
        last, gap = np.nan, 0
        for t in range(x.size):
            if np.isnan(z[t]):
                gap += 1
                continue
            pred = 0.0 if np.isnan(last) else (self.phi_ ** (gap + 1)) * last
            out[t] = (z[t] - pred) / self.sd_e_
            last, gap = z[t], 0
        return out

    def score(self, values) -> np.ndarray:
        e = self.innovations(values)
        sp = sm = 0.0
        out = np.zeros(e.size)
        for t in range(e.size):
            if np.isnan(e[t]):
                continue
            sp = max(0.0, sp + e[t] - self.k)
            sm = max(0.0, sm - e[t] - self.k)
            out[t] = max(sp, sm)
        return out
