import numpy as np
import pytest

from anomaly_lite.stream_data import weekend_shape
from anomaly_lite.streaming import OnlineHarmonicProfile

WEEK = 672


def test_int_and_tuple_agree_and_column_count():
    t = np.arange(3000)
    a = OnlineHarmonicProfile((96, WEEK), 2)
    b = OnlineHarmonicProfile((96, WEEK), (2, 2))
    assert np.allclose(a._design(t), b._design(t))
    assert a.n_harmonics == 2
    c = OnlineHarmonicProfile((96, WEEK), (1, 3))
    assert c._design(t).shape[1] == 1 + 2 * 1 + 2 * 3
    assert c.harmonics == (1, 3)


def test_more_weekly_harmonics_fit_the_weekend_dip_better():
    rng = np.random.default_rng(0)
    n = 6000
    t = np.arange(n)
    x = 2 * np.sin(2 * np.pi * t / 96) + 1.5 * weekend_shape(t, WEEK) + 0.3 * rng.normal(size=n)
    fit = slice(0, 3000)
    r = {}
    for h in ((1, 1), (1, 3)):
        m = OnlineHarmonicProfile((96, WEEK), h, 0.995).fit(x[fit], t[fit])
        r[h] = np.nanstd(m.transform(x[3000:], t[3000:]))
    assert r[(1, 3)] < 0.85 * r[(1, 1)]


def test_bad_harmonic_spec():
    with pytest.raises(ValueError):
        OnlineHarmonicProfile((96, WEEK), (1, 2, 3))
    with pytest.raises(ValueError):
        OnlineHarmonicProfile((96,), (0,))
