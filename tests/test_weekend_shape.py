import numpy as np
import pytest

from anomaly_lite.stream_data import make_drifting_stream, weekend_shape

WEEK = 7 * 96


def test_zero_mean_unit_half_range_and_dipped_weekend():
    w = weekend_shape(np.arange(WEEK), WEEK)
    assert abs(w.mean()) < 1e-12
    assert np.isclose(w.max() - w.min(), 2.0)
    day = WEEK // 7
    assert w[day // 2: 5 * day - day // 2].min() > 0.4
    assert w[5 * day + day // 2: 7 * day - day // 2].max() < -1.0


def test_periodic_and_not_a_sinusoid():
    t = np.arange(3 * WEEK)
    w = weekend_shape(t, WEEK)
    assert np.allclose(w[:WEEK], w[WEEK:2 * WEEK])
    spec = np.abs(np.fft.rfft(w[:WEEK]))
    assert spec[2] + spec[3] > 0.3 * spec[1]  # real power beyond the first harmonic


def test_ramps_are_short_and_smooth():
    w = weekend_shape(np.arange(WEEK), WEEK)
    jumps = np.abs(np.diff(np.r_[w, w[0]]))
    assert jumps.max() < 0.2


def test_default_stream_is_the_slice4_stream():
    a = make_drifting_stream(n=3000, n_events=9, warmup=600, min_gap=80, random_state=3, season2_period=WEEK, season2_amp=1.5)
    b = make_drifting_stream(n=3000, n_events=9, warmup=600, min_gap=80, random_state=3, season2_period=WEEK, season2_amp=1.5, season2_shape="sine")
    assert np.allclose(a.values, b.values)


def test_weekend_stream_differs_only_by_the_shape():
    a = make_drifting_stream(n=3000, n_events=9, warmup=600, min_gap=80, random_state=3, season2_period=WEEK, season2_amp=1.5)
    b = make_drifting_stream(n=3000, n_events=9, warmup=600, min_gap=80, random_state=3, season2_period=WEEK, season2_amp=1.5, season2_shape="weekend")
    t = np.arange(3000)
    d = b.values - a.values
    expected = 1.5 * (weekend_shape(t, WEEK) - np.sin(2 * np.pi * t / WEEK + 1.0))
    assert np.allclose(d, expected)
    assert (a.labels == b.labels).all() and b.meta["season2_shape"] == "weekend"


def test_validation():
    with pytest.raises(ValueError):
        make_drifting_stream(n=500, season2_period=WEEK, season2_amp=1.0, season2_shape="square")
    with pytest.raises(ValueError):
        weekend_shape(np.arange(10), 5)
