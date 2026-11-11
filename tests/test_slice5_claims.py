"""Reduced-scale (4 evaluation seeds, static threshold) re-check of each directional slice-5 claim."""

import pytest

from anomaly_lite.slice5_eval import run_slice5_grid


@pytest.fixture(scope="module")
def f1():
    g = run_slice5_grid(seeds=range(4), scenarios=("weekly", "weekend"), protocols=("static_quantile",))
    return g.pivot_table(index=["scenario", "seed"], columns=["deseason", "detector"], values="event_f1")


def test_h1_weekly_harmonics_fix_the_weekend_dip_for_robust_z(f1):
    we = f1.loc["weekend"]
    assert (we[("harm_dw3", "robust_z")] > we[("harm_dw", "robust_z")]).all()
    wk = f1.loc["weekly"]
    assert wk[("harm_dw3", "robust_z")].mean() >= wk[("harm_dw", "robust_z")].mean() - 0.01


def test_prewhitening_rescues_textbook_k(f1):
    for d in ("harm_dw", "harm_dw3"):
        assert (f1[(d, "cusum_pw")] > f1[(d, "cusum")] + 0.3).all()


def test_prewhitened_cusum_survives_a_misfit_deseasonaliser(f1):
    we = f1.loc["weekend"]
    assert (we[("harm_dw", "cusum_pw")] > we[("harm_dw", "cusum_k15")]).all()
    assert abs(we[("harm_dw", "cusum_pw")].mean() - we[("harm_dw3", "cusum_pw")].mean()) < 0.05


def test_h2_failure_reproduced_tuned_k15_wins_on_a_good_fit(f1):
    wk = f1.loc["weekly"]
    assert wk[("harm_dw3", "cusum_pw")].mean() < wk[("harm_dw3", "cusum_k15")].mean()


def test_robust_z_still_beats_any_cusum_with_the_recommended_profile(f1):
    for sc in ("weekly", "weekend"):
        x = f1.loc[sc]
        assert x[("harm_dw3", "robust_z")].mean() > x[("harm_dw3", "cusum_pw")].mean()
        assert x[("harm_dw3", "robust_z")].mean() > x[("harm_dw3", "cusum_k15")].mean()
