import numpy as np

from anomaly_lite.slice4_eval import residuals4
from anomaly_lite.slice5_eval import SCENARIOS5, residuals5
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream
from anomaly_lite.streaming import RECOMMENDED_WEEKLY_HARMONICS, OnlineHarmonicProfile, weekly_harmonic_profile


def test_factory_is_the_preregistered_h1_profile():
    m = weekly_harmonic_profile()
    assert isinstance(m, OnlineHarmonicProfile)
    assert m.periods == (96, 672) and m.harmonics == RECOMMENDED_WEEKLY_HARMONICS == (1, 3)
    assert m.forgetting == 0.995 and m.missing == "skip"


def test_harmonics_1_1_reproduces_slice4_harm_dw():
    st = make_drifting_stream(n=4000, n_events=14, warmup=600, min_gap=80, random_state=102, **SCENARIOS5["weekly"])
    fit, _, _ = contiguous_windows(4000, (0.3, 0.2, 0.5))
    t, x = st.frame["t"].to_numpy(), st.values
    ref = residuals4(x, t, fit, 96, "harm_dw")
    rest = slice(fit.stop, None)
    got = weekly_harmonic_profile(harmonics=(1, 1)).fit(x[fit], t[fit]).transform(x[rest], t[rest])
    assert np.allclose(ref[rest], got, equal_nan=True)


def test_slice5_harness_uses_the_factory():
    st = make_drifting_stream(n=4000, n_events=14, warmup=600, min_gap=80, random_state=103, **SCENARIOS5["weekend"])
    fit, _, _ = contiguous_windows(4000, (0.3, 0.2, 0.5))
    t, x = st.frame["t"].to_numpy(), st.values
    rest = slice(fit.stop, None)
    a = residuals5(x, t, fit, 96, "harm_dw3")[rest]
    b = weekly_harmonic_profile().fit(x[fit], t[fit]).transform(x[rest], t[rest])
    assert np.allclose(a, b, equal_nan=True)
