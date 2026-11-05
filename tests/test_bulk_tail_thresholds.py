"""bulk_tail_threshold / rolling_bulk_tail_thresholds: contamination-robust label-free thresholds (slice 4)."""

import numpy as np
import pytest

from anomaly_lite.thresholds import (
    apply_thresholds,
    bulk_tail_threshold,
    false_alarm_threshold,
    rolling_bulk_tail_thresholds,
)


def test_exponential_tail_matches_the_closed_form_on_exponential_data():
    s = np.random.default_rng(0).exponential(2.0, 200_000)
    thr = bulk_tail_threshold(s, rate=0.005, tail="exponential")
    assert thr == pytest.approx(2.0 * np.log(1 / 0.005), rel=0.02)  # true 99.5% quantile 10.6
    assert np.mean(s > thr) == pytest.approx(0.005, abs=0.0015)


def test_halfnormal_tail_matches_abs_gaussian():
    s = np.abs(np.random.default_rng(1).normal(0, 1.5, 200_000))
    thr = bulk_tail_threshold(s, rate=0.005, tail="halfnormal")
    assert thr == pytest.approx(1.5 * 2.807, rel=0.02)  # Phi^-1(0.9975) = 2.807
    assert np.mean(s > thr) == pytest.approx(0.005, abs=0.0015)


def test_five_percent_contamination_pushes_the_plain_rank_up_but_not_the_bulk_fit():
    rng = np.random.default_rng(2)
    normal = np.abs(rng.normal(0, 1, 1140))
    bad = rng.uniform(6, 12, 60)  # 5% anomalies far above the normal bulk
    s = rng.permutation(np.r_[normal, bad])
    plain = false_alarm_threshold(s, 0.005)
    bulk = bulk_tail_threshold(s, 0.005, tail="halfnormal")
    assert plain > 5.9  # inside the anomalies: would never alarm on a normal-score excursion
    assert 2.4 < bulk < 3.4  # close to the clean 2.81, and below every anomaly
    assert (bad > bulk).all()


def test_rolling_layout_nan_before_start_refresh_and_causality():
    rng = np.random.default_rng(3)
    s = np.abs(rng.normal(size=4000))
    thr = rolling_bulk_tail_thresholds(s, 0.005, window=1200, start=1500, step=50, tail="halfnormal")
    assert np.isnan(thr[:1500]).all() and np.isfinite(thr[1500:]).all()
    assert (thr[1500:1550] == thr[1500]).all()
    s2 = s.copy()
    s2[3000:] *= 10  # the future must not move earlier thresholds
    thr2 = rolling_bulk_tail_thresholds(s2, 0.005, window=1200, start=1500, step=50, tail="halfnormal")
    np.testing.assert_array_equal(thr[:3000], thr2[:3000])
    assert thr2[3990] > 5 * thr[3990]  # window now 80%+ rescaled data


def test_rolling_tracks_a_scale_drift_and_valid_mask_drops_placeholders():
    rng = np.random.default_rng(4)
    s = np.abs(rng.normal(size=6000)) * np.linspace(1, 3, 6000)
    thr = rolling_bulk_tail_thresholds(s, 0.005, window=1200, start=1200, step=50, tail="halfnormal")
    assert thr[-1] / thr[1300] == pytest.approx(2.6, rel=0.25)
    miss = rng.random(6000) < 0.2
    s0 = np.where(miss, 0.0, s)
    a = rolling_bulk_tail_thresholds(s0, 0.005, window=1200, start=1200, valid=~miss, tail="halfnormal")
    b = rolling_bulk_tail_thresholds(s0, 0.005, window=1200, start=1200, tail="halfnormal")
    assert np.nanmean(a) > 1.05 * np.nanmean(b)  # zero placeholders (20%) drag the bulk quantiles down
    assert apply_thresholds(s, thr)[:1200].sum() == 0


def test_validation_and_small_sample():
    assert bulk_tail_threshold(np.arange(30.0)) == float("inf")
    with pytest.raises(ValueError):
        bulk_tail_threshold(np.arange(500.0), rate=0.2)
    with pytest.raises(ValueError):
        bulk_tail_threshold(np.arange(500.0), tail="gamma")
    with pytest.raises(ValueError):
        rolling_bulk_tail_thresholds(np.arange(500.0), window=10)
