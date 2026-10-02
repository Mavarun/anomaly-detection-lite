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
