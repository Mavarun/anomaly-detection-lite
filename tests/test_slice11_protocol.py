"""Slice 11's pre-registered rules and decision function, fixed before the evaluation seeds ran."""

import pandas as pd

from anomaly_lite import slice11_eval as S


def test_protocol_fixed():
    assert S.EVAL_SEEDS == tuple(range(700, 710)) and not set(S.EVAL_SEEDS) & set(S.DEV_SEEDS)
    assert S.ARMS == {"f995": 0.995, "f999": 0.999} and S.FAR_CAP == 0.0075
    assert set(S.PREREGISTERED) == {"H1_forgetting", "H2_guardrail", "expectation", "reported_not_decided"}
    used = {600 + i for i in range(10)} | {300 + i for i in range(10)}
    assert not set(S.EVAL_SEEDS) & used  # slice 7 and slice 10 seeds


def _df(t3, g, far=0.005):
    rows = []
    for sc, pairs in (("weekend_t3", t3), ("weekend", g)):
        for i, (a, b) in enumerate(pairs):
            rows.append({"scenario": sc, "seed": i, "f995_f1": a, "f999_f1": b, "f995_far": far, "f999_far": far})
    return pd.DataFrame(rows)


def test_decide_passes_and_each_failure():
    good = [(0.70, 0.72)] * 7 + [(0.70, 0.69)] * 3
    gauss = [(0.80, 0.80)] * 10
    r = S.decide(_df(good, gauss))
    assert r["H1"] and r["arm"] == "f999" and r["t3_seeds_better"] == 7
    assert not S.decide(_df([(0.70, 0.75)] * 6 + [(0.70, 0.69)] * 4, gauss))["H1"]
    assert not S.decide(_df(good, [(0.80, 0.78)] * 10))["H1"]
    assert not S.decide(_df(good, gauss, far=0.008))["H1"]


def test_one_cell_runs():
    row = S.run_slice11_experiment(150, "weekend_t3")
    assert round(row["f995_f1"], 3) == 0.723 and round(row["f999_f1"], 3) == 0.735
