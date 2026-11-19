"""Slice 8 protocol: oracle threshold, fresh seeds, the rule on a toy grid."""

import numpy as np
import pandas as pd

from anomaly_lite import slice6_eval as S6, slice7_eval as S7, slice8_eval as S8


def test_seeds_are_fresh():
    used = set(S6.EVAL_SEEDS) | set(S6.DEV_SEEDS) | set(S7.EVAL_SEEDS) | set(S7.DEV_SEEDS)
    assert not set(S8.EVAL_SEEDS) & used


def test_oracle_finds_the_separating_threshold():
    rng = np.random.default_rng(0)
    y = np.zeros(2000, int)
    for k in range(10):
        y[100 + 150 * k: 105 + 150 * k] = 1
    s = rng.normal(size=2000) + 6.0 * y
    thr, q, f1 = S8.oracle_threshold(s, y, np.ones(2000, bool))
    assert f1 == 1.0 and 2.0 < thr < 6.0 and 0.9 <= q < 1.0


def test_a_monotone_transform_flags_the_same_points():
    # the reason slice 7's "rank or log transform" suggestion cannot help a static threshold
    s = np.abs(np.random.default_rng(1).standard_t(3, size=5000))
    thr = np.quantile(s, 0.995)
    assert np.array_equal(s > thr, np.log1p(s) > np.log1p(thr))


def _toy(oracle_t3, static_t3=0.70, static_g=0.85):
    rows = []
    for s in range(10):
        rows.append({"seed": s, "scenario": "weekend", "detector": "robust_z", "static_f1": static_g, "oracle_f1": 0.9})
        rows.append({"seed": s, "scenario": "weekend_t3", "detector": "robust_z", "static_f1": static_t3,
                     "oracle_f1": oracle_t3 + 0.001 * s})
    return pd.DataFrame(rows)


def test_rule():
    d = S8.decide(_toy(0.80))
    assert d["H1"] and not d["H2_exhausted"]
    d = S8.decide(_toy(0.72))
    assert not d["H1"] and d["H2_exhausted"]
