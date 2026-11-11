import numpy as np

from anomaly_lite.slice5_eval import (
    DEV_SEEDS, EVAL_SEEDS, PREREGISTERED, SCENARIOS5, decide, residuals5, run_slice5_experiment, run_slice5_grid,
)
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream


def test_preregistration_is_fixed_and_seeds_disjoint():
    assert set(PREREGISTERED) == {"H1_weekly_harmonics", "H2_prewhitened_cusum", "reported_not_decided"}
    assert ">= 7 of 10" in PREREGISTERED["H1_weekly_harmonics"]
    assert "1.05" in PREREGISTERED["H2_prewhitened_cusum"]
    assert not set(EVAL_SEEDS) & set(DEV_SEEDS)


def test_harm_dw3_residual_is_causal_and_static_on_fit():
    st = make_drifting_stream(n=4000, n_events=14, warmup=600, min_gap=80, random_state=101, **SCENARIOS5["weekend"])
    fit, _, _ = contiguous_windows(4000, (0.3, 0.2, 0.5))
    t, x = st.frame["t"].to_numpy(), st.values
    a = residuals5(x, t, fit, st.meta["period"], "harm_dw3")
    b = residuals5(x, t, fit, st.meta["period"], "harm_dw")
    assert np.allclose(a[fit], b[fit])
    x2 = x.copy()
    x2[3000:] += 50.0
    c = residuals5(x2, t, fit, st.meta["period"], "harm_dw3")
    assert np.allclose(a[:3000], c[:3000])


def test_experiment_rows_and_decide_shape_on_dev_seeds():
    g = run_slice5_grid(seeds=DEV_SEEDS[:2], scenarios=("weekly", "weekend"), n=4000, n_events=14,
                        protocols=("static_quantile",))
    assert len(g) == 2 * 2 * 2 * 5
    assert g.event_f1.between(0, 1).all()
    d = decide(g)
    assert set(d) >= {"H1", "H2", "recommended_deseason"}


def test_single_run_has_both_protocols():
    r = run_slice5_experiment(seed=100, scenario="weekend_missing", n=4000, n_events=14, detectors=("cusum_pw",))
    assert set(r.protocol) == {"static_quantile", "rolling_bulk"} and set(r.deseason) == {"harm_dw", "harm_dw3"}
