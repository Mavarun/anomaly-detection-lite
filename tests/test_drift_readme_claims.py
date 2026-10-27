"""Reduced-scale (4 seeds) re-check of every directional claim in the slice-3 README."""

import pytest

from anomaly_lite.drift_eval import run_drift_grid

SEEDS = range(4)


@pytest.fixture(scope="module")
def m():
    long = run_drift_grid(seeds=SEEDS, scenarios=("stationary", "drift", "drift_missing"))
    return long.groupby(["scenario", "deseason", "protocol", "detector"])[
        ["event_f1", "point_far", "cost_per_1k"]].mean()


def f1(m, sc, ds, pr, det):
    return m.loc[(sc, ds, pr, det), "event_f1"]


def test_drift_hurts_fit_once_and_online_recovers(m):
    for det in ("robust_z", "ewma"):
        assert f1(m, "drift", "static", "static_quantile", det) < f1(m, "stationary", "static", "static_quantile", det)
        assert f1(m, "drift", "online", "static_quantile", det) > f1(m, "drift", "static", "static_quantile", det)
    assert m.loc[("drift", "online", "static_quantile", "robust_z"), "cost_per_1k"] < \
        m.loc[("drift", "static", "static_quantile", "robust_z"), "cost_per_1k"]
    gap = abs(f1(m, "stationary", "online", "static_quantile", "robust_z")
              - f1(m, "stationary", "static", "static_quantile", "robust_z"))
    assert gap < 0.03


def test_cusum_is_unusable_under_drift_either_way(m):
    assert m.loc[("drift", "static", "static_quantile", "cusum"), "point_far"] > 0.5
    assert f1(m, "drift", "online", "static_quantile", "cusum") < 0.4


def test_label_free_rolling_quantile_is_far_too_conservative(m):
    for det in ("robust_z", "ewma", "variance"):
        assert m.loc[("drift", "online", "rolling_quantile", det), "point_far"] < 0.002
    assert f1(m, "drift", "online", "rolling_quantile", "robust_z") < \
        f1(m, "drift", "online", "static_quantile", "robust_z") - 0.15
    assert abs(f1(m, "drift", "online", "rolling_rescaled", "robust_z")
               - f1(m, "drift", "online", "static_quantile", "robust_z")) < 0.03


def test_variance_union_does_not_beat_robust_z_and_missing_data_costs_little(m):
    assert f1(m, "drift", "online", "static_quantile", "variance") < 0.65
    assert f1(m, "drift", "online", "static_quantile", "rz_or_var") <= \
        f1(m, "drift", "online", "static_quantile", "robust_z") + 0.01
    d = f1(m, "drift", "online", "static_quantile", "robust_z") - \
        f1(m, "drift_missing", "online", "static_quantile", "robust_z")
    assert -0.01 < d < 0.05
