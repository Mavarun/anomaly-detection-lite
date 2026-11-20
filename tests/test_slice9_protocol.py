"""Slice 9 protocol is frozen, and the pooled rule really uses more calibration points."""

import numpy as np
import pandas as pd

from anomaly_lite import slice9_eval as S


def test_frozen_constants():
    assert S.EVAL_SEEDS == tuple(range(500, 510)) and S.DEV_SEEDS == (150,)
    assert not set(S.DEV_SEEDS) & set(S.EVAL_SEEDS)
    assert (S.RATE, S.FAR_CAP) == (0.005, 0.0075)
    assert set(S.PREREGISTERED) == {"H1_pooled_calibration", "H2_guardrail", "expectation", "reported_not_decided"}
    assert "seeds 500-509" in S.PREREGISTERED["H1_pooled_calibration"]


def test_dev_seed_runs_and_pools_more_points():
    df = S.run_slice9_experiment(S.DEV_SEEDS[0], "weekend_t3")
    assert list(df.detector) == list(S.DETECTORS9)
    assert (df.n_pooled > 2 * df.n_cal).all()
    for c in ("quantile_far", "pooled_far", "quantile_f1", "pooled_f1"):
        assert df[c].between(0, 1).all()


def _fake(d3, dg, far):
    rows = []
    for i, (a, b) in enumerate(zip(d3, dg)):
        rows.append({"scenario": "weekend_t3", "seed": i, "detector": "robust_z", "quantile_f1": 0.7,
                     "pooled_f1": 0.7 + a, "quantile_far": 0.005, "pooled_far": far})
        rows.append({"scenario": "weekend", "seed": i, "detector": "robust_z", "quantile_f1": 0.85,
                     "pooled_f1": 0.85 + b, "quantile_far": 0.005, "pooled_far": 0.005})
    return pd.DataFrame(rows)


def test_decide_passes_only_when_all_three_hold():
    good = S.decide(_fake([0.02] * 8 + [-0.01] * 2, [0.0] * 10, 0.006))
    assert good["H1"] and good["rule"] == "pooled" and good["t3_seeds_better"] == 8
    assert not S.decide(_fake([0.02] * 6 + [-0.01] * 4, [0.0] * 10, 0.006))["H1"]      # 6/10 seeds
    assert not S.decide(_fake([0.02] * 8 + [-0.01] * 2, [-0.02] * 10, 0.006))["H1"]    # Gaussian cost
    bad_far = S.decide(_fake([0.02] * 8 + [-0.01] * 2, [0.0] * 10, 0.009))              # FAR cap
    assert not bad_far["H1"] and bad_far["rule"] == "quantile"
    assert np.isclose(bad_far["guardrail_margin"], 0.7 - 0.75)
