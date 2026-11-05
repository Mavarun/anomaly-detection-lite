"""OnlineHarmonicProfile: robust-RLS level + sinusoids (slice 4)."""

import numpy as np
import pytest

from anomaly_lite.streaming import OnlineHarmonicProfile, OnlineSeasonalProfile


def _series(n=4000, noise=0.3, seed=0, weekly=1.5, growth=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    amp = 2.0 * (1 + growth * t / n)
    x = 10 + amp * np.sin(2 * np.pi * t / 96) + weekly * np.sin(2 * np.pi * t / 672 + 1.0)
    return t, x + rng.normal(0, noise, n)


def test_fit_recovers_known_coefficients():
    t, x = _series(n=3000, noise=0.05, weekly=1.5)
    m = OnlineHarmonicProfile(periods=(96, 672)).fit(x, t)
    th = m.theta_  # [b0, sin96, cos96, sin672, cos672]
    assert th[0] == pytest.approx(10, abs=0.05)
    assert th[1] == pytest.approx(2.0, abs=0.05) and th[2] == pytest.approx(0.0, abs=0.05)
    assert np.hypot(th[3], th[4]) == pytest.approx(1.5, abs=0.07)


def test_second_period_removed_where_single_period_profile_fails():
    t, x = _series(n=6000, noise=0.3)
    fit, rest = slice(0, 1800), slice(1800, 6000)
    one = OnlineSeasonalProfile(96).fit(x[fit], t[fit]).transform(x[rest], t[rest])
    two = OnlineHarmonicProfile(periods=(96, 672)).fit(x[fit], t[fit]).transform(x[rest], t[rest])
    late = slice(1000, None)
    assert np.std(one[late]) > 0.55
    assert np.std(two[late]) < 0.40  # noise sd 0.3: the leftover is the learning cost
    assert abs(np.mean(two[late])) < 0.05


def test_tracks_growing_amplitude_and_level_drift():
    t, x = _series(n=6000, noise=0.3, weekly=0.0, growth=1.0)
    x = x + 3.0 * t / 6000
    fit, rest = slice(0, 1800), slice(1800, 6000)
    r = OnlineHarmonicProfile(periods=(96,)).fit(x[fit], t[fit]).transform(x[rest], t[rest])
    # a ramp of 5e-4/step is followed with a lag of ~memory x slope = 0.1; fit-once would be off by > 1.5
    assert np.std(r[-1000:]) < 0.42 and 0 < np.mean(r[-1000:]) < 0.15


def test_causal_future_rewrite_does_not_move_past_residuals():
    t, x = _series(n=3000)
    m = OnlineHarmonicProfile(periods=(96, 672)).fit(x[:1200], t[:1200])
    a = m.transform(x[1200:], t[1200:])
    y = x.copy()
    y[2300:] += 50.0
    b = m.transform(y[1200:], t[1200:])
    np.testing.assert_array_equal(a[:1100], b[:1100])


def test_spike_barely_moves_state_and_long_shift_absorbed_slowly():
    t, x = _series(n=3600, noise=0.3, weekly=0.0)
    m = OnlineHarmonicProfile(periods=(96,)).fit(x[:1200], t[:1200])
    base = m.transform(x[1200:], t[1200:])
    y = x.copy()
    y[2000] += 30.0
    spiked = m.transform(y[1200:], t[1200:])
    assert spiked[2000 - 1200] > 25  # the spike is in the residual in full
    assert np.max(np.abs(spiked[2010 - 1200:] - base[2010 - 1200:])) < 0.35
    z = x.copy()
    z[2000:2040] += 3.0  # 10 sd, 40 steps
    sh = m.transform(z[1200:], t[1200:])
    assert np.mean(sh[2030 - 1200:2040 - 1200]) > 2.0  # not absorbed within the event


def test_missing_points_skip_and_validation():
    t, x = _series(n=2500)
    x2 = x.copy()
    x2[1500:1520] = np.nan
    m = OnlineHarmonicProfile(periods=(96, 672), missing="skip").fit(x2[:1200], t[:1200])
    r = m.transform(x2[1200:], t[1200:])
    assert np.isnan(r[300:320]).all() and np.isfinite(r[:300]).all() and np.isfinite(r[320:]).all()
    with pytest.raises(RuntimeError):
        OnlineHarmonicProfile().transform(x)
    with pytest.raises(ValueError):
        OnlineHarmonicProfile(periods=(1,))
    with pytest.raises(ValueError):
        OnlineHarmonicProfile(forgetting=0.5)
    with pytest.raises(ValueError):
        OnlineHarmonicProfile(periods=(96, 672)).fit(x[:10], t[:10])
