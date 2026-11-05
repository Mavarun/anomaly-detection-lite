"""FrozenReferenceCUSUM: Page CUSUM whose reference adapts while calm and freezes while alarmed (slice 4)."""

import numpy as np
import pytest

from anomaly_lite.streaming import CUSUMDetector, FrozenReferenceCUSUM


def _noise(n=3000, seed=0, sd=1.0):
    return np.random.default_rng(seed).normal(0, sd, n)


def test_slow_ramp_does_not_blow_up_fixed_reference_cusum_does():
    x = _noise(6000) + 4.0 * np.arange(6000) / 6000  # 4 sd of ramp
    fixed = CUSUMDetector(k=0.5, reference=500).score(x)
    frozen = FrozenReferenceCUSUM(k=0.5, reference=500).score(x)
    assert np.mean(fixed[2000:] > 5) > 0.5
    # the reference lags a ramp by ~0.13 sd, which eats part of the allowance k: some excess alarms remain
    assert np.mean(frozen[2000:] > 5) < 0.08
    assert np.mean(frozen[2000:] > 5) < np.mean(fixed[2000:] > 5) / 8


def test_freezing_keeps_a_moderate_shift_visible_that_a_following_reference_absorbs():
    x = _noise(3000, seed=1)
    x[1500:1560] += 1.2
    frozen = FrozenReferenceCUSUM(k=0.5, reference=500, freeze_at=2.0).score(x)
    follow = FrozenReferenceCUSUM(k=0.5, reference=500, freeze_at=1e9).score(x)  # never freezes
    assert frozen[1500:1560].max() > follow[1500:1560].max() + 1.0
    assert frozen[1500:1560].max() > 5.0


def test_permanent_shift_alarms_then_reanchors_and_goes_quiet():
    x = _noise(3000, seed=2)
    x[1500:] += 3.0
    d = FrozenReferenceCUSUM(k=0.5, reference=500, max_freeze=150, anchor=30)
    sc = d.score(x)
    assert sc[1500:1560].max() > 10 and d.n_reanchor_ >= 1
    assert np.mean(sc[2200:] > 5) < 0.05  # the new level became the reference


def test_quiet_stream_scores_stay_low_and_scale_is_floored():
    x = _noise(5000, seed=3)
    sc = FrozenReferenceCUSUM(k=0.5, reference=500).score(x)
    assert np.mean(sc[500:] > 8) < 0.01
    d = FrozenReferenceCUSUM(k=0.5, reference=500)
    d.score(np.r_[x[:500], np.full(4000, x[:500].mean())])  # constant tail must not crash (scale floored)


def test_causal_and_gaps_hold_state():
    x = _noise(3000, seed=4)
    d = FrozenReferenceCUSUM(k=0.5, reference=500, missing="skip")
    a = d.score(x)
    y = x.copy()
    y[2500:] += 40.0
    np.testing.assert_array_equal(a[:2500], d.score(y)[:2500])
    g = x.copy()
    g[1000:1030] = np.nan
    s = d.score(g)
    assert (s[1000:1030] == 0).all() and np.isfinite(s).all()
    with pytest.raises(ValueError):
        FrozenReferenceCUSUM(max_freeze=0)
    with pytest.raises(ValueError):
        FrozenReferenceCUSUM(missing="bad").score(x)
