"""Tests for the end-to-end streaming evaluation and the Nile CUSUM demo."""

import numpy as np
import pytest

from anomaly_lite.stream_eval import (
    load_nile,
    nile_cusum_alarms,
    run_streaming_experiment,
    run_streaming_grid,
    summarise_streaming,
)


@pytest.fixture(scope="module")
def one_seed():
    return run_streaming_experiment(seed=0)


def test_experiment_structure(one_seed):
    df = one_seed
    assert len(df) == 4 * 3
    assert set(df["detector"]) == {"ewma", "cusum", "robust_z", "random"}
    assert set(df["protocol"]) == {"cal_quantile", "cal_best_f1", "test_oracle"}
    for col in ("event_f1", "pa_f1", "point_f1", "point_roc_auc", "point_far"):
        assert df[col].between(0, 1).all()
    assert (df["n_test_events"] >= 8).all()


def test_random_control_behaves_like_chance(one_seed):
    r = one_seed[one_seed.detector == "random"]
    assert r["point_roc_auc"].iloc[0] == pytest.approx(0.5, abs=0.05)


def test_robust_z_beats_random_on_event_f1_with_calibrated_threshold():
    long = run_streaming_grid(seeds=range(3))
    q = long[long.protocol == "cal_quantile"].groupby("detector")["event_f1"].mean()
    assert q["robust_z"] > q["random"] + 0.3
    s = summarise_streaming(long, ["event_f1", "pa_f1"])
    assert {"event_f1_mean", "pa_f1_std"} <= set(s.columns)


def test_point_adjust_flatters_random_control():
    long = run_streaming_grid(seeds=range(3))
    r = long[(long.detector == "random") & (long.protocol == "cal_best_f1")]
    assert r["pa_f1"].mean() > r["event_f1"].mean() + 0.1


def test_nile_loader_and_cusum_alarm_near_known_break():
    df = load_nile()
    assert len(df) == 100 and df["year"].iloc[0] == 1871 and df["year"].iloc[-1] == 1970
    alarms = nile_cusum_alarms(references=(20,), thresholds=(5.0,))
    year = alarms["first_alarm_year"].iloc[0]
    assert 1898 <= year <= 1905  # textbook level drop ~1898-1899


def test_nile_alarm_is_sensitive_to_reference_window():
    alarms = nile_cusum_alarms(references=(20, 25), thresholds=(5.0,)).set_index(
        "reference_years"
    )
    # a 25-year reference ending in 1895 gives a tighter scale and alarms early
    assert alarms.loc[25, "first_alarm_year"] < 1898 <= alarms.loc[20, "first_alarm_year"]
