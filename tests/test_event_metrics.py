"""Tests for point, point-adjusted and event-level alarm metrics."""

import numpy as np
import pytest

from anomaly_lite.event_metrics import (
    all_alarm_metrics,
    event_metrics,
    point_adjust,
    point_adjusted_metrics,
    point_metrics,
    segments,
)


def test_segments():
    assert segments([0, 1, 1, 0, 0, 1, 0, 1]) == [(1, 2), (5, 5), (7, 7)]
    assert segments([0, 0]) == []
    assert segments([1, 1]) == [(0, 1)]


def test_single_alarm_in_long_event_point_vs_pa_vs_event():
    y = np.zeros(1000, dtype=int)
    y[100:200] = 1
    pred = np.zeros_like(y)
    pred[150] = 1
    pm = point_metrics(y, pred)
    assert pm.precision == 1.0 and pm.recall == pytest.approx(0.01)
    pa = point_adjusted_metrics(y, pred)
    assert pa.recall == 1.0 and pa.f1 == 1.0
    ev = event_metrics(y, pred)
    assert ev.recall == 1.0 and ev.precision == 1.0 and ev.mean_delay == 50


def test_point_adjust_only_fills_hit_events():
    y = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 1])
    np.testing.assert_array_equal(point_adjust(y, pred), [0, 1, 1, 1, 0, 0, 0, 1])


def test_random_alarms_inflate_point_adjusted_f1():
    """Kim et al. (2022) critique: PA rewards random alarms on long events."""
    rng = np.random.default_rng(0)
    y = np.zeros(10_000, dtype=int)
    for s in range(500, 10_000, 1000):
        y[s : s + 80] = 1  # 10 long events, prevalence 8%
    pred = (rng.uniform(size=y.size) < 0.05).astype(int)
    pf1 = point_metrics(y, pred).f1
    paf1 = point_adjusted_metrics(y, pred).f1
    ev = event_metrics(y, pred)
    assert pf1 < 0.1
    assert paf1 > 0.5  # looks "good" despite pure noise
    assert ev.recall == 1.0 and ev.precision < 0.15  # event precision exposes it


def test_tolerance_counts_lagged_alarms_but_not_early_ones():
    y = np.zeros(100, dtype=int)
    y[40:50] = 1
    late = np.zeros_like(y)
    late[53] = 1
    early = np.zeros_like(y)
    early[37] = 1
    assert event_metrics(y, late, tolerance=0).recall == 0.0
    assert event_metrics(y, late, tolerance=0).n_false_segments == 1
    r = event_metrics(y, late, tolerance=5)
    assert r.recall == 1.0 and r.n_false_segments == 0 and r.mean_delay == 13
    assert event_metrics(y, early, tolerance=5).recall == 0.0


def test_false_alarm_rate_and_no_alarms():
    y = np.zeros(2000, dtype=int)
    y[1000:1010] = 1
    pred = np.zeros_like(y)
    pred[[100, 101, 500, 1500]] = 1  # 3 false segments
    r = event_metrics(y, pred)
    assert r.n_pred_segments == 3 and r.n_false_segments == 3
    assert r.false_alarms_per_1k == pytest.approx(3000 / 1990)
    none = event_metrics(y, np.zeros_like(y))
    assert none.f1 == 0.0 and np.isnan(none.mean_delay)


def test_all_alarm_metrics_keys_and_validation():
    y = np.array([0, 1, 1, 0])
    d = all_alarm_metrics(y, np.array([0, 1, 0, 0]))
    assert {"point_f1", "pa_f1", "event_f1", "event_mean_delay"} <= set(d)
    with pytest.raises(ValueError):
        point_metrics([0, 2], [0, 1])
    with pytest.raises(ValueError):
        event_metrics([0, 1], [0, 1, 0])
    with pytest.raises(ValueError):
        event_metrics([0, 1], [0, 1], tolerance=-1)
