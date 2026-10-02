"""Tests for causal streaming detectors."""

import numpy as np
import pytest

from anomaly_lite.streaming import (
    CUSUMDetector,
    EWMADetector,
    RobustZDetector,
    SeasonalProfile,
)

DETECTORS = [
    EWMADetector(alpha=0.05, warmup=30),
    CUSUMDetector(k=0.5, reference=100),
    RobustZDetector(window=50),
]


def _noise(n=1000, seed=0):
    return np.random.default_rng(seed).normal(size=n)


@pytest.mark.parametrize("det", DETECTORS, ids=lambda d: type(d).__name__)
def test_scores_are_causal(det):
    x = _noise(800, 1)
    base = det.score(x)
    y = x.copy()
    y[500:] += np.random.default_rng(9).normal(0, 5, size=300)  # rewrite the future
    pert = det.score(y)
    np.testing.assert_allclose(base[:500], pert[:500])
    assert base.shape == x.shape and np.all(base >= 0)


@pytest.mark.parametrize("det", DETECTORS, ids=lambda d: type(d).__name__)
def test_isolated_spike_scores_far_above_background(det):
    x = _noise(1000, 2)
    x[600] += 8.0
    s = det.score(x)
    assert s[600] > np.quantile(s[200:590], 0.99)


def test_cusum_accumulates_small_level_shift_better_than_robust_z():
    x = _noise(1200, 3)
    x[700:800] += 1.0  # 1-sd persistent shift
    cus = CUSUMDetector(k=0.5, reference=500).score(x)
    rz = RobustZDetector(window=200).score(x)
    assert cus[799] > 15.0  # ~ (1 - 0.5) * 100 minus noise
    assert np.median(rz[700:800]) < 2.0
    # but CUSUM keeps alarming after the shift ends (lagged recovery)
    assert cus[810] > 5.0


def test_cusum_reset_scheme_restarts_sums():
    x = _noise(600, 4)
    x[300:] += 2.0
    s = CUSUMDetector(k=0.5, reference=200, reset_threshold=5.0).score(x)
    assert s.max() < 5.0 + 4.0  # never runs away when reset
    assert (s[300:] > 5.0).sum() >= 10  # repeated alarms during the shift


def test_ewma_clipping_limits_spike_contamination():
    """A huge spike must not mask a moderate anomaly 20 steps later."""
    x = _noise(1000, 5)
    x[500] += 50.0
    x[520] += 5.0
    clipped = EWMADetector(alpha=0.05, clip=3.0).score(x)
    unclipped = EWMADetector(alpha=0.05, clip=1e9).score(x)
    assert clipped[520] > 3.0
    assert unclipped[520] < 0.5 * clipped[520]  # variance inflated by the spike
    assert np.median(clipped[530:600]) < 1.2


def test_robust_z_excludes_current_point_and_handles_zero_mad():
    x = np.zeros(100)
    x[60] = 1.0
    s = RobustZDetector(window=20).score(x)
    assert s[60] > 1e6  # MAD == 0 -> tiny floor -> huge but finite
    assert np.isfinite(s).all()
    assert s[59] == 0.0


def test_seasonal_profile_removes_fixed_pattern():
    rng = np.random.default_rng(6)
    t = np.arange(3000)
    x = 5 + 3 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 0.3, size=t.size)
    prof = SeasonalProfile(period=48).fit(x[:960], t[:960])
    resid = prof.transform(x, t)
    assert np.std(resid[960:]) < 0.4
    with pytest.raises(ValueError):
        SeasonalProfile(period=48).fit(x[:10], t[:10])


def test_validation():
    with pytest.raises(ValueError):
        EWMADetector(alpha=1.5)
    with pytest.raises(ValueError):
        CUSUMDetector(k=-1)
    with pytest.raises(ValueError):
        RobustZDetector(window=2)
    with pytest.raises(ValueError):
        EWMADetector().score(np.array([1.0, np.nan]))
