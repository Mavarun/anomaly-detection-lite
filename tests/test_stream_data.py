"""Tests for the labelled synthetic stream generator."""

import numpy as np
import pytest

from anomaly_lite.stream_data import contiguous_windows, make_labelled_stream


def test_shapes_and_reproducibility():
    a = make_labelled_stream(n=3000, n_events=9, random_state=4)
    b = make_labelled_stream(n=3000, n_events=9, random_state=4)
    assert list(a.frame.columns) == ["t", "value", "label", "event_id"]
    assert len(a.frame) == 3000
    np.testing.assert_array_equal(a.values, b.values)
    assert a.events == b.events


def test_events_disjoint_gapped_and_match_labels():
    s = make_labelled_stream(n=6000, n_events=18, min_gap=120, warmup=1000, random_state=1)
    ev = sorted(s.events, key=lambda e: e["start"])
    for e1, e2 in zip(ev[:-1], ev[1:]):
        assert e2["start"] > e1["end"] + 120
    assert all(e["start"] >= 1000 for e in ev)
    assert s.labels[:1000].sum() == 0
    assert s.labels.sum() == sum(e["length"] for e in ev)
    for e in ev:
        seg = s.frame.iloc[e["start"] : e["end"] + 1]
        assert (seg["label"] == 1).all() and (seg["event_id"] == e["event_id"]).all()
    assert {e["type"] for e in ev} == {"spike", "level_shift", "variance_burst"}


def test_spikes_are_large_relative_to_noise():
    s = make_labelled_stream(n=6000, n_events=18, season_amp=0.0, random_state=2)
    sd = s.meta["marginal_sd"]
    for e in (e for e in s.events if e["type"] == "spike"):
        dev = np.abs(s.values[e["start"] : e["end"] + 1] - 10.0).max()
        assert dev > 2.5 * sd


def test_variance_burst_increases_local_std():
    s = make_labelled_stream(n=6000, n_events=18, season_amp=0.0, random_state=3)
    base = np.std(s.values[:800])
    for e in (e for e in s.events if e["type"] == "variance_burst"):
        seg = s.values[e["start"] : e["end"] + 1]
        assert np.std(seg) > 1.5 * base


def test_contiguous_windows():
    w = contiguous_windows(1000, (0.3, 0.2, 0.5))
    assert [(x.start, x.stop) for x in w] == [(0, 300), (300, 500), (500, 1000)]
    with pytest.raises(ValueError):
        contiguous_windows(100, (0.5, 0.6))


def test_bad_arguments():
    with pytest.raises(ValueError):
        make_labelled_stream(ar_phi=1.0)
    with pytest.raises(ValueError):
        make_labelled_stream(event_types=("unicorn",))
    with pytest.raises(ValueError):
        make_labelled_stream(n=500, n_events=50, warmup=100)


def test_drifting_stream_without_drift_or_gaps_is_the_slice2_stream():
    from anomaly_lite.stream_data import make_drifting_stream, make_labelled_stream

    a = make_labelled_stream(n=3000, n_events=9, random_state=4)
    b = make_drifting_stream(n=3000, n_events=9, random_state=4, trend_sd=0, walk_sd=0, amp_growth=0)
    assert np.allclose(a.values, b.values)
    assert (a.labels == b.labels).all() and a.events == b.events
    assert b.frame["observed"].all()


def test_drift_moves_the_late_level_and_amplitude_but_not_the_labels():
    from anomaly_lite.stream_data import make_drifting_stream, make_labelled_stream

    a = make_labelled_stream(n=6000, random_state=1)
    b = make_drifting_stream(n=6000, random_state=1, trend_sd=3.0, walk_sd=0.0, amp_growth=1.0)
    d = b.values - a.values
    sd = a.meta["marginal_sd"]
    assert abs(d[:200].mean()) < 0.2 * sd
    assert d[-500:].mean() > 2.0 * sd  # the trend
    late = slice(5000, 6000)
    assert np.ptp(b.values[late] - a.values[late] - d[late].mean()) > 2.0  # amplitude grew
    assert (a.labels == b.labels).all()


def test_missing_gaps_hit_the_target_fraction_and_spare_event_starts():
    from anomaly_lite.stream_data import make_drifting_stream

    s = make_drifting_stream(n=6000, random_state=2, missing_frac=0.05)
    obs = s.frame["observed"].to_numpy()
    assert 0.05 <= 1 - obs.mean() < 0.06
    assert np.isnan(s.values[~obs]).all() and np.isfinite(s.values[obs]).all()
    assert all(obs[e["start"]] for e in s.events)
    assert obs[: s.meta["warmup"]].all()
    with pytest.raises(ValueError):
        make_drifting_stream(missing_frac=0.6)
