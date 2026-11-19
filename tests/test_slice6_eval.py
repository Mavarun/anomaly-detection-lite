import numpy as np

from anomaly_lite.slice6_eval import (
    CLIP, DETECTORS6, DEV_SEEDS, EVAL_SEEDS, PREREGISTERED, SCENARIOS6, decide, run_slice6_experiment,
    run_slice6_grid,
)


def test_preregistration_is_fixed_and_seeds_disjoint():
    assert set(PREREGISTERED) == {"H1_clipped_cusum", "H2_shipped_robust_z_guardrail", "reported_not_decided"}
    assert ">= 7 of 10" in PREREGISTERED["H1_clipped_cusum"] and "0.10" in PREREGISTERED["H2_shipped_robust_z_guardrail"]
    assert CLIP == 3.0
    assert not set(EVAL_SEEDS) & set(DEV_SEEDS)


def test_gaussian_control_matches_slice5_stream():
    assert "noise_df" not in SCENARIOS6["weekend"]
    assert SCENARIOS6["weekend_t3"]["noise_df"] == 3.0


def test_rows_and_decide_shape_on_dev_seeds():
    g = run_slice6_grid(seeds=DEV_SEEDS[:2], n=4000, n_events=14)
    assert len(g) == 2 * len(SCENARIOS6) * len(DETECTORS6)
    assert g.event_f1.between(0, 1).all()
    d = decide(g)
    assert set(d) >= {"H1", "H1_parts", "H2", "recommended_clip"}
    assert d["recommended_clip"] in (None, CLIP)


def test_single_run_is_deterministic():
    a = run_slice6_experiment(seed=101, scenario="weekend_t5", n=4000, n_events=14)
    b = run_slice6_experiment(seed=101, scenario="weekend_t5", n=4000, n_events=14)
    np.testing.assert_array_equal(a.event_f1.to_numpy(), b.event_f1.to_numpy())
