"""Reduced-scale (4 seeds) re-check of every directional claim in the slice-4 README."""

import pytest

from anomaly_lite.slice4_eval import run_slice4_grid

SEEDS = range(4)


@pytest.fixture(scope="module")
def m():
    long = run_slice4_grid(seeds=SEEDS, scenarios=("drift", "weekly"), deseason=("online", "harm_dw"))
    return long.groupby(["scenario", "deseason", "protocol", "detector"])[["event_f1", "point_far", "cost_per_1k"]].mean()


def f1(m, sc, ds, pr, det):
    return m.loc[(sc, ds, pr, det), "event_f1"]


def test_weekly_cycle_hurts_the_one_period_profile_and_harmonic_terms_fix_it(m):
    a, b = f1(m, "weekly", "online", "static_quantile", "robust_z"), f1(m, "weekly", "harm_dw", "static_quantile", "robust_z")
    assert b > a + 0.04
    assert m.loc[("weekly", "harm_dw", "static_quantile", "robust_z"), "cost_per_1k"] < \
        m.loc[("weekly", "online", "static_quantile", "robust_z"), "cost_per_1k"] - 3


def test_extra_harmonic_terms_cost_nothing_measurable_without_a_weekly_cycle(m):
    gap = f1(m, "drift", "harm_dw", "static_quantile", "robust_z") - f1(m, "drift", "online", "static_quantile", "robust_z")
    assert abs(gap) < 0.04


def test_bulk_tail_rolling_threshold_repairs_the_plain_rolling_rank_for_z_scores(m):
    for det in ("robust_z", "ewma"):
        bulk = f1(m, "weekly", "harm_dw", "rolling_bulk", det)
        assert bulk > f1(m, "weekly", "harm_dw", "rolling_quantile", det) + 0.05
        assert abs(bulk - f1(m, "weekly", "harm_dw", "static_quantile", det)) < 0.05
    assert m.loc[("weekly", "harm_dw", "rolling_bulk", "robust_z"), "point_far"] < 0.006
    assert m.loc[("weekly", "harm_dw", "rolling_quantile", "robust_z"), "point_far"] < 0.002


def test_bulk_tail_model_hurts_the_variance_detector(m):
    assert f1(m, "weekly", "harm_dw", "rolling_bulk", "variance") < \
        f1(m, "weekly", "harm_dw", "static_quantile", "variance") - 0.05


def test_cusum_allowance_matters_more_than_the_reference_scheme(m):
    k15 = f1(m, "weekly", "harm_dw", "static_quantile", "cusum_k15")
    assert k15 > f1(m, "weekly", "harm_dw", "static_quantile", "cusum") + 0.3
    assert f1(m, "weekly", "harm_dw", "static_quantile", "cusum_frozen_k15") < k15 - 0.05  # freezing does not help
    assert f1(m, "weekly", "harm_dw", "static_quantile", "robust_z") > k15 - 0.02  # robust z still at least as good
