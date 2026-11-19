import numpy as np
import pytest
from scipy import stats

from anomaly_lite.stream_data import make_drifting_stream, make_labelled_stream


def test_default_noise_reproduces_gaussian_stream_exactly():
    a = make_labelled_stream(n=6000, random_state=3)
    b = make_labelled_stream(n=6000, random_state=3, noise_df=0)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.events == b.events


def test_student_t_keeps_events_and_labels_but_changes_noise():
    g = make_labelled_stream(n=6000, random_state=5)
    t = make_labelled_stream(n=6000, random_state=5, noise_df=3)
    assert g.events == t.events
    np.testing.assert_array_equal(g.labels, t.labels)
    assert not np.allclose(g.values, t.values)
    assert t.meta["noise_df"] == 3.0


def test_student_t_innovations_have_unit_variance_and_heavy_tails():
    # Strip the deterministic season / level and the AR(1) to recover the innovations.
    s = make_labelled_stream(n=40000, random_state=11, noise_df=4, n_events=0, warmup=10, min_gap=10,
                             season_amp=0.0)
    x = s.values - 10.0  # default level
    e = (x[1:] - s.meta["ar_phi"] * x[:-1]) / s.meta["noise_sd"]
    assert abs(e.std() - 1.0) < 0.05
    assert stats.kurtosis(e) > 1.5  # Gaussian is 0; t(4) has infinite theoretical kurtosis


@pytest.mark.parametrize("df", [-1, 1, 2])
def test_invalid_df_rejected(df):
    with pytest.raises(ValueError):
        make_labelled_stream(n=500, noise_df=df)


def test_drifting_stream_passes_noise_df_through():
    a = make_drifting_stream(n=6000, random_state=2, noise_df=3)
    assert a.meta["noise_df"] == 3.0
