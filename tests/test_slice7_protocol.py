"""Slice 7 protocol: fresh seeds, the rule on a toy grid and a dev-seed smoke run."""

import pandas as pd

from anomaly_lite import slice6_eval as S6
from anomaly_lite import slice7_eval as S7


def test_seeds_are_fresh():
    assert not set(S7.EVAL_SEEDS) & (set(S6.EVAL_SEEDS) | set(S6.DEV_SEEDS))


def _toy(gain_t3, gain_g, cost_t3=(10.0, 9.0), f1=(0.8, 0.7)):
    rows = []
    for s in range(10):
        for sc, gain, base in (("weekend", gain_g, f1[0]), ("weekend_t3", gain_t3, f1[1])):
            for proto, add in (("static_quantile", 0.0), ("static_gpd", gain)):
                c = cost_t3[0] if proto == "static_quantile" else cost_t3[1]
                rows.append({"seed": s, "scenario": sc, "deseason": "harm_dw3", "protocol": proto,
                             "detector": "robust_z", "event_f1": base + add + 0.001 * s, "cost_per_1k": c})
    return pd.DataFrame(rows)


def test_rule():
    d = S7.decide(_toy(0.05, 0.0))
    assert d["H1"] and d["chosen"] == "static_gpd" and d["H2"]
    assert not S7.decide(_toy(0.05, -0.02))["H1"]
    assert not S7.decide(_toy(0.05, 0.0, cost_t3=(9.0, 10.0)))["H1"]
    assert not S7.decide(_toy(0.05, 0.0, f1=(0.9, 0.7)))["H2"]


def test_dev_smoke():
    df = S7.run_slice7_experiment(S7.DEV_SEEDS[0], "weekend_t3")
    assert set(df.protocol) == set(S7.PROTOCOLS7) and set(df.detector) == set(S7.DETECTORS7)
    assert df.threshold.notna().all()


def test_posthoc_reading_drops_one_seed():
    df = pd.concat([S7.run_slice7_experiment(s, "weekend") for s in S7.DEV_SEEDS[:2]], ignore_index=True)
    full = S7.posthoc_threshold_reading(df)
    one = S7.posthoc_threshold_reading(df, drop_seed=S7.DEV_SEEDS[1])
    assert len(full) == len(one) == 6
    assert one.threshold_sd.isna().all() and full.threshold_sd.notna().all()
