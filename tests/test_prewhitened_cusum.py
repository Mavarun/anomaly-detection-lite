import numpy as np
import pytest

from anomaly_lite.streaming import CUSUMDetector, PrewhitenedCUSUM


def ar1(n, phi, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_recovers_phi_and_whitens():
    x = ar1(4000, 0.6, 0)
    d = PrewhitenedCUSUM(reference=2000)
    e = d.innovations(x)
    assert abs(d.phi_ - 0.6) < 0.05
    r1 = np.corrcoef(e[1:], e[:-1])[0, 1]
    assert abs(r1) < 0.05
    assert abs(np.std(e[2000:]) - 1.0) < 0.1


def test_null_score_far_smaller_than_raw_k05_on_ar1():
    x = ar1(6000, 0.6, 1)
    raw = CUSUMDetector(k=0.5, reference=2000).score(x)[2000:]
    pw = PrewhitenedCUSUM(k=0.5, reference=2000).score(x)[2000:]
    assert np.quantile(pw, 0.995) < 0.5 * np.quantile(raw, 0.995)


def test_detects_level_shift():
    x = ar1(4000, 0.6, 2)
    x[3000:] += 2.0 * np.std(x[:2000])
    s = PrewhitenedCUSUM(k=0.5, reference=2000).score(x)
    thr = np.quantile(s[2000:3000], 0.999)
    first = np.argmax(s[3000:] > thr)
    assert s[3000:].max() > thr and first < 60


def test_iid_input_is_left_almost_unchanged():
    x = np.random.default_rng(3).normal(size=3000)
    d = PrewhitenedCUSUM(reference=1500)
    e = d.innovations(x)
    assert abs(d.phi_) < 0.06
    assert np.corrcoef(e, (x - d.mu_) / d.sd_)[0, 1] > 0.99


def test_missing_points_held_and_gap_uses_phi_power():
    x = ar1(1000, 0.6, 4)
    x[500:510] = np.nan
    d = PrewhitenedCUSUM(reference=400, missing="skip")
    e = d.innovations(x)
    assert np.isnan(e[500:510]).all() and np.isfinite(e[510])
    z = (x - d.mu_) / d.sd_
    assert np.isclose(e[510], (z[510] - d.phi_ ** 11 * z[499]) / d.sd_e_)
    s = d.score(x)
    assert (s[500:510] == 0).all()


def test_validation():
    with pytest.raises(ValueError):
        PrewhitenedCUSUM(k=-1)
    with pytest.raises(ValueError):
        PrewhitenedCUSUM(reference=5).score(np.arange(5.0))
