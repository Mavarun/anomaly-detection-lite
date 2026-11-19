"""gpd_tail_threshold: peaks-over-threshold quantile for heavy-tailed calibration scores (slice 7)."""

import numpy as np
import pytest
from scipy.stats import norm, t

from anomaly_lite.thresholds import false_alarm_threshold, gpd_tail_threshold


def test_close_to_the_true_quantile_on_large_gaussian_samples():
    s = np.abs(np.random.default_rng(0).normal(size=20000))
    assert gpd_tail_threshold(s, 0.005) == pytest.approx(norm.ppf(1 - 0.0025), abs=0.08)


def test_less_variable_than_the_empirical_rank_under_t3():
    rng = np.random.default_rng(1)
    emp, gpd = [], []
    for _ in range(200):
        s = np.abs(rng.standard_t(3, size=1200))
        emp.append(false_alarm_threshold(s, 0.005))
        gpd.append(gpd_tail_threshold(s, 0.005))
    truth = t.ppf(1 - 0.0025, 3)
    assert np.std(gpd) < np.std(emp)
    assert abs(np.median(gpd) - truth) < 0.15 * truth


def test_falls_back_with_few_exceedances_and_validates():
    s = np.arange(100.0)
    assert gpd_tail_threshold(s, 0.005, tail_frac=0.1, min_exceed=30) == false_alarm_threshold(s, 0.005)
    with pytest.raises(ValueError):
        gpd_tail_threshold(s, 0.2, tail_frac=0.1)
    with pytest.raises(ValueError):
        gpd_tail_threshold([np.nan], 0.005)


def test_shape_clip_bounds_a_wild_tail():
    s = np.concatenate([np.abs(np.random.default_rng(2).normal(size=1000)), [1e6, 2e6, 5e6]])
    assert np.isfinite(gpd_tail_threshold(s, 0.005)) and gpd_tail_threshold(s, 0.005) < 1e4
