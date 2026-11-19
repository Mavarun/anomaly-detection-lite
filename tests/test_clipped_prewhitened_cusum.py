import numpy as np
import pytest

from anomaly_lite.streaming import PrewhitenedCUSUM


def _ar1(n, phi=0.6, seed=0, df=None):
    rng = np.random.default_rng(seed)
    eps = rng.standard_t(df, n) * np.sqrt((df - 2) / df) if df else rng.normal(size=n)
    x = np.empty(n)
    prev = 0.0
    for t in range(n):
        prev = phi * prev + eps[t]
        x[t] = prev
    return x


def test_default_clip_none_is_unchanged():
    x = _ar1(1500, seed=1)
    a = PrewhitenedCUSUM(k=0.5, reference=500).score(x)
    b = PrewhitenedCUSUM(k=0.5, reference=500, clip=None).score(x)
    np.testing.assert_array_equal(a, b)


def test_clip_bounds_a_single_spike():
    x = _ar1(1500, seed=2)
    x[1000] += 30.0
    raw = PrewhitenedCUSUM(k=0.5, reference=500).score(x)
    clipped = PrewhitenedCUSUM(k=0.5, reference=500, clip=3.0).score(x)
    assert raw[1000] - raw[999] > 10
    assert clipped[1000] - clipped[999] <= 3.0 - 0.5 + 1e-9


def test_clip_keeps_a_sustained_shift():
    x = _ar1(1500, seed=3)
    x[1000:1100] += 2.0 * np.std(x[:500])
    c = PrewhitenedCUSUM(k=0.5, reference=500, clip=3.0).score(x)
    assert c[1099] > 20


def test_clip_reduces_null_peaks_under_t3_noise():
    peaks_raw, peaks_clip = [], []
    for s in range(5):
        x = _ar1(3000, seed=10 + s, df=3)
        peaks_raw.append(PrewhitenedCUSUM(k=0.5, reference=1000).score(x)[1000:].max())
        peaks_clip.append(PrewhitenedCUSUM(k=0.5, reference=1000, clip=3.0).score(x)[1000:].max())
    assert np.mean(peaks_clip) < np.mean(peaks_raw)


def test_clip_must_exceed_k():
    with pytest.raises(ValueError):
        PrewhitenedCUSUM(k=0.5, clip=0.5)
