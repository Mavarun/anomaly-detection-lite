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
