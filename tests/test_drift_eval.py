"""Slice-3 harness: structure, leakage barriers and the paired-effect arithmetic."""

import numpy as np
import pandas as pd
import pytest

from anomaly_lite.drift_eval import (
    PROTOCOLS,
    SCENARIOS,
    paired_effects,
    residuals,
    run_drift_experiment,
    summarise,
)
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream


@pytest.fixture(scope="module")
def one():
    return run_drift_experiment(seed=0, scenario="drift_missing")


def test_structure_and_metric_ranges(one):
    assert set(one.protocol) == set(PROTOCOLS) and set(one.deseason) == {"static", "online"}
    assert set(one.detector) == {"ewma", "cusum", "robust_z", "variance", "random", "rz_or_var"}
    assert len(one) == 2 * len(PROTOCOLS) * 6
    assert one.event_f1.between(0, 1).all() and one.point_far.between(0, 1).all()
    assert (one.cost_per_1k >= 0).all() and (one.n_test_events > 8).all()


def test_online_residuals_are_causal_and_equal_static_on_the_fit_window():
    s = make_drifting_stream(n=6000, n_events=24, warmup=600, min_gap=80, random_state=1, **SCENARIOS["drift"])
    fit, _, _ = contiguous_windows(6000, (0.3, 0.2, 0.5))
    t, x = s.frame["t"].to_numpy(), s.values
    a = residuals(x, t, fit, 96, "online")
    assert np.allclose(a[fit], residuals(x, t, fit, 96, "static")[fit])
    y = x.copy()
    y[4000:] += 20.0
    assert np.allclose(a[:4000], residuals(y, t, fit, 96, "online")[:4000])
    with pytest.raises(ValueError):
        residuals(x, t, fit, 96, "magic")


def test_test_labels_do_not_move_any_threshold(monkeypatch):
    import anomaly_lite.drift_eval as D

    base = run_drift_experiment(seed=2, scenario="drift", deseason=("online",))
    orig = D.make_drifting_stream

    def flipped(**kw):
        st = orig(**kw)
        lab = st.frame["label"].to_numpy().copy()
        lab[3000:] = 0  # erase every test-window label
        st.frame["label"] = lab
        return st

    monkeypatch.setattr(D, "make_drifting_stream", flipped)
    erased = run_drift_experiment(seed=2, scenario="drift", deseason=("online",))
    # thresholds come from fit / cal only, so every detector raises the same alarms
    assert (erased.event_recall == 0).all() and base.event_recall.max() > 0.5
    assert np.array_equal(base.n_alarm_points.to_numpy(), erased.n_alarm_points.to_numpy())


def test_random_control_is_poor_and_robust_z_beats_it(one):
    f1 = one.groupby("detector").event_f1.mean()
    assert f1["random"] < 0.3 and f1["robust_z"] > f1["random"] + 0.4


def test_paired_effects_arithmetic():
    long = pd.DataFrame({
        "seed": [0, 1, 2, 0, 1, 2], "scenario": ["s"] * 6, "deseason": ["a"] * 3 + ["b"] * 3,
        "protocol": ["p"] * 6, "detector": ["d"] * 6, "event_f1": [0.5, 0.6, 0.7, 0.6, 0.6, 0.9],
    })
    e = paired_effects(long, "deseason", "a", "b", "event_f1").iloc[0]
    assert e.diff_mean == pytest.approx((0.1 + 0.0 + 0.2) / 3)
    assert (e.n_b_better, e.n_ties, e.n_seeds) == (2, 1, 3)
    assert e.sign_p == pytest.approx(0.5)  # 2 of 2 non-ties
    assert e.ci_lo <= e.diff_mean <= e.ci_hi
    s = summarise(long, ["event_f1"])
    assert list(s.event_f1_mean.round(3)) == [0.6, 0.7]
