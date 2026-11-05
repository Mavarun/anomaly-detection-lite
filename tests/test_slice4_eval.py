"""Slice-4 harness: structure, label hygiene, causality of the pipeline and a reduced-scale sanity check."""

import numpy as np
import pytest

from anomaly_lite.slice4_eval import (
    DESEASON4,
    DETECTORS4,
    PROTOCOLS4,
    SCENARIOS4,
    alarms4,
    residuals4,
    run_slice4_experiment,
    score4,
)
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream


def test_long_frame_structure_and_reproducibility():
    a = run_slice4_experiment(seed=1, scenario="weekly", n=3000, n_events=12)
    b = run_slice4_experiment(seed=1, scenario="weekly", n=3000, n_events=12)
    assert len(a) == len(DESEASON4) * len(PROTOCOLS4) * len(DETECTORS4)
    assert set(a.detector) == set(DETECTORS4) and set(a.scenario) == {"weekly"}
    assert a["event_f1"].between(0, 1).all() and (a["point_far"].dropna().between(0, 1)).all()
    assert a.equals(b)


def test_scenarios_have_the_intended_drift_and_second_cycle():
    assert SCENARIOS4["drift"].get("season2_period", 0) == 0
    assert SCENARIOS4["weekly"]["season2_period"] == 672 and SCENARIOS4["weekly_missing"]["missing_frac"] == 0.05


def test_test_labels_cannot_move_any_alarm():
    st = make_drifting_stream(n=4000, n_events=14, warmup=600, min_gap=80, random_state=3, **SCENARIOS4["weekly"])
    fit, cal, test = contiguous_windows(4000)
    t, x, y, obs = st.frame["t"].to_numpy(), st.values, st.labels, st.frame["observed"].to_numpy()
    resid = residuals4(x, t, fit, 96, "harm_dw")
    scores = score4(resid, fit.stop)
    y2 = y.copy()
    y2[test] = 1 - y2[test]  # rewrite every test label
    for proto in PROTOCOLS4:
        for det in ("robust_z", "cusum_frozen_k15"):
            np.testing.assert_array_equal(alarms4(scores[det], y, obs, cal, test, proto, det),
                                          alarms4(scores[det], y2, obs, cal, test, proto, det))


def test_residuals_after_fit_window_do_not_see_the_future():
    st = make_drifting_stream(n=4000, n_events=14, warmup=600, min_gap=80, random_state=5, **SCENARIOS4["weekly"])
    fit, _, _ = contiguous_windows(4000)
    t, x = st.frame["t"].to_numpy(), st.values
    y = x.copy()
    y[3500:] += 100.0
    for how in DESEASON4:
        a, b = residuals4(x, t, fit, 96, how), residuals4(y, t, fit, 96, how)
        np.testing.assert_array_equal(a[:3500], b[:3500])
    with pytest.raises(ValueError):
        residuals4(x, t, fit, 96, "nope")


def test_second_cycle_hurts_one_period_profile_and_harmonic_fixes_it():
    rows = [run_slice4_experiment(seed=s, scenario="weekly", n=6000, n_events=24, deseason=("online", "harm_dw"),
                                  protocols=("static_quantile",)) for s in range(3)]
    import pandas as pd

    m = pd.concat(rows).groupby(["deseason", "detector"])["event_f1"].mean()
    assert m[("harm_dw", "robust_z")] > m[("online", "robust_z")] + 0.03
    assert m[("harm_dw", "robust_z")] > 0.8
