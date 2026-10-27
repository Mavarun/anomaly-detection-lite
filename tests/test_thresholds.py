"""Tests for held-out threshold calibration protocols."""

import numpy as np
import pytest

from anomaly_lite.thresholds import (
    apply_threshold,
    best_f1_threshold,
    evaluate_threshold_protocols,
    false_alarm_threshold,
)


def test_false_alarm_threshold_rank():
    s = np.arange(1, 101, dtype=float)  # n = 100
    # k = ceil(101 * 0.95) = 96 -> 96th smallest
    assert false_alarm_threshold(s, rate=0.05) == 96.0
    assert false_alarm_threshold(np.arange(10.0), rate=0.01) == np.inf


def test_false_alarm_threshold_drops_labelled_anomalies():
    s = np.r_[np.zeros(99), 1000.0]
    lab = np.r_[np.zeros(99), 1]
    assert false_alarm_threshold(s, 0.05, labels=lab) == 0.0


def test_realised_far_matches_target_on_iid_scores():
    rng = np.random.default_rng(0)
    fars = []
    for _ in range(200):
        t = false_alarm_threshold(rng.normal(size=500), rate=0.02)
        fars.append(apply_threshold(rng.normal(size=2000), t).mean())
    assert np.mean(fars) == pytest.approx(0.02, abs=0.003)
    assert np.mean(fars) <= 0.02 + 0.002  # conservative on average


def test_best_f1_threshold_separates_toy():
    s = np.r_[np.linspace(0, 1, 90), np.linspace(5, 6, 10)]
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
    t, f = best_f1_threshold(s, y, metric="point_f1", min_quantile=0.0)
    assert 1.0 <= t < 5.0 and f == pytest.approx(1.0)


def _stream(seed):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=3000)
    y = np.zeros(3000, dtype=int)
    for start in range(200, 3000, 300):
        y[start : start + 20] = 1
        s[start : start + 20] += rng.uniform(1.5, 4.0)
    return s, y


def test_protocols_use_only_calibration_window():
    s, y = _stream(1)
    cal, test = slice(0, 1500), slice(1500, 3000)
    a = evaluate_threshold_protocols(s, y, cal, test, rate=0.01)
    y2 = y.copy()
    y2[1500:] = 0
    y2[2000:2050] = 1  # rewrite test labels
    b = evaluate_threshold_protocols(s, y2, cal, test, rate=0.01)
    ta, tb = a.set_index("protocol").threshold, b.set_index("protocol").threshold
    assert ta["cal_quantile"] == tb["cal_quantile"]
    assert ta["cal_best_f1"] == tb["cal_best_f1"]
    assert ta["test_oracle"] != tb["test_oracle"]


def test_oracle_is_upper_reference_for_its_metric():
    for seed in range(5):
        s, y = _stream(seed)
        r = evaluate_threshold_protocols(s, y, slice(0, 1500), slice(1500, 3000)).set_index(
            "protocol"
        )
        assert r.loc["test_oracle", "event_f1"] >= r.loc["cal_best_f1", "event_f1"] - 1e-12
        assert r.loc["test_oracle", "event_f1"] >= r.loc["cal_quantile", "event_f1"] - 1e-12


def test_validation():
    with pytest.raises(ValueError):
        false_alarm_threshold(np.ones(10), rate=0.0)
    with pytest.raises(ValueError):
        best_f1_threshold(np.ones(5), np.zeros(5), metric="auc")


# --- walk-forward thresholds (slice 3) ----------------------------------------

def test_rolling_thresholds_only_use_the_past_and_no_labels():
    from anomaly_lite.thresholds import rolling_false_alarm_thresholds

    rng = np.random.default_rng(0)
    s = rng.normal(size=3000)
    thr = rolling_false_alarm_thresholds(s, rate=0.01, window=500, start=1000, step=25)
    assert np.isnan(thr[:1000]).all() and np.isfinite(thr[1000:]).all()
    s2 = s.copy()
    s2[2000:] += 10.0  # rewrite the future
    thr2 = rolling_false_alarm_thresholds(s2, rate=0.01, window=500, start=1000, step=25)
    assert np.array_equal(thr[:2000], thr2[:2000], equal_nan=True)
    # piecewise constant between refreshes, and equal to the static rule on its window
    assert len(np.unique(thr[1000:1025])) == 1
    from anomaly_lite.thresholds import false_alarm_threshold
    assert thr[1500] == false_alarm_threshold(s[1000:1500], 0.01)


def test_rolling_thresholds_track_a_drifting_score_scale_where_a_static_one_fails():
    from anomaly_lite.thresholds import apply_thresholds, false_alarm_threshold, rolling_false_alarm_thresholds

    rng = np.random.default_rng(1)
    n = 6000
    s = np.abs(rng.normal(size=n)) * np.linspace(1.0, 2.0, n)  # scale drifts up x2
    static = false_alarm_threshold(s[1000:2000], 0.005)
    far_static = (s[3000:] > static).mean()
    roll = rolling_false_alarm_thresholds(s, 0.005, window=1000, start=2000, step=50)
    far_roll = apply_thresholds(s, roll)[3000:].mean()
    assert far_static > 0.03  # >6x the target
    assert far_roll < 0.012


def test_valid_mask_and_apply_thresholds():
    from anomaly_lite.thresholds import apply_thresholds, rolling_false_alarm_thresholds

    s = np.r_[np.zeros(100), np.arange(100.0)]
    v = np.r_[np.zeros(100, dtype=bool), np.ones(100, dtype=bool)]
    a = rolling_false_alarm_thresholds(s, 0.05, window=200, start=200, step=1, valid=v)
    assert a.size == 200 and np.isnan(a).all()  # start == n: nothing to threshold
    thr = rolling_false_alarm_thresholds(np.r_[s, s], 0.05, window=200, start=200, step=1,
                                         valid=np.r_[v, v])
    assert thr[200] >= 95  # the placeholder zeros were dropped
    assert apply_thresholds([1.0, 5.0, 2.0], [np.nan, 4.0, 3.0]).tolist() == [0, 1, 0]
    with pytest.raises(ValueError):
        rolling_false_alarm_thresholds(s, 0.05, window=5)
