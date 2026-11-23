import numpy as np
import pandas as pd

from anomaly_lite import slice10_eval as S
from anomaly_lite.slice5_eval import residuals5
from anomaly_lite.slice6_eval import SCENARIOS6, score6
from anomaly_lite.stream_data import contiguous_windows, make_drifting_stream


def _stream(seed=150, sc="weekend"):
    st = make_drifting_stream(n=6000, n_events=24, warmup=600, min_gap=80, random_state=seed, **SCENARIOS6[sc])
    return st, contiguous_windows(6000, (0.3, 0.2, 0.5))


def test_frozen_design():
    assert S.EVAL_SEEDS == tuple(range(600, 610)) and not set(S.EVAL_SEEDS) & set(S.DEV_SEEDS)
    assert (S.RATE, S.FAR_CAP, S.WARMUP, S.MECHANISM_CAP) == (0.005, 0.0075, 200, 0.01)
    assert set(S.PREREGISTERED) >= {"C0_mechanism", "H1_pooled_weekly", "H2_guardrail", "expectation"}


def test_weekly_fit_scores_change_only_the_fit_window_residual():
    st, (fit, cal, test) = _stream()
    t, x = st.frame["t"].to_numpy(), st.values
    shipped = score6(residuals5(x, t, fit, st.meta["period"], "harm_dw3"), fit.stop - fit.start)
    weekly = S.weekly_fit_scores(x, t, fit, st.meta["period"])
    # robust_z is a 200-point rolling score: past the fit window plus its window, it is unchanged
    a, b = shipped["robust_z"][fit.stop + 200:], weekly["robust_z"][fit.stop + 200:]
    assert np.allclose(a, b, equal_nan=True)
    assert not np.allclose(shipped["robust_z"][fit][300:], weekly["robust_z"][fit][300:], equal_nan=True)


def test_shipped_fit_window_still_holds_the_weekend_dip():
    st, (fit, _, _) = _stream(151)
    t, x = st.frame["t"].to_numpy(), st.values
    r = residuals5(x, t, fit, st.meta["period"], "harm_dw3")[fit]
    from anomaly_lite.slice4_eval import WEEK
    from anomaly_lite.streaming import weekly_harmonic_profile
    w = weekly_harmonic_profile(st.meta["period"], WEEK).fit(x[fit], t[fit]).transform(x[fit], t[fit])
    assert np.nanstd(r) > 1.3 * np.nanstd(w)


def test_rules_on_hand_made_rows():
    rows = []
    for sc, gain in (("weekend", 0.0), ("weekend_t3", 0.05)):
        for s in range(10):
            rows.append({"scenario": sc, "seed": s, "detector": "robust_z", "quantile_f1": 0.7,
                         "pooled_weekly_f1": 0.7 + gain, "quantile_far": 0.005, "pooled_weekly_far": 0.006,
                         "fit_exceed_rate": 0.004})
    d = S.decide(pd.DataFrame(rows))
    assert d["H1"] and d["C0"] and d["rule"] == "pooled_weekly" and d["H2"]
    df = pd.DataFrame(rows)
    df.loc[df.scenario == "weekend_t3", "pooled_weekly_far"] = 0.01
    assert not S.decide(df)["H1"]
