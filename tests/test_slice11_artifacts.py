"""Slice 11's committed artifacts recompute from one seed, and the decision follows from the grid."""

import json
from pathlib import Path

import pandas as pd

from anomaly_lite import slice11_eval as S

ART = Path(__file__).resolve().parents[1] / "artifacts"
DF = pd.read_csv(ART / "slice11_forgetting.csv")
DEC = json.loads((ART / "slice11_decisions.json").read_text())


def test_grid_shape_and_decision():
    assert len(DF) == 30 and set(DF.seed) == set(S.EVAL_SEEDS)
    assert DEC["preregistered"] == S.PREREGISTERED
    got = S.decide(DF)
    for k, v in DEC["decision"].items():
        assert got[k] == v if isinstance(v, (bool, str, int)) else abs(got[k] - v) < 1e-5, k


def test_reading():
    d = DEC["decision"]
    assert not d["H1"] and d["t3_seeds_better"] == 4 and d["arm"] == "f995" and not d["H2"]
    assert 0 < d["t3_f1_diff"] < 0.01 and d["gauss_f1_diff"] > -0.01 and d["t3_f999_far"] <= S.FAR_CAP


def test_one_seed_recomputes():
    row = S.run_slice11_experiment(705, "weekend_t3")
    ref = DF[(DF.seed == 705) & (DF.scenario == "weekend_t3")].iloc[0]
    assert abs(row["f999_f1"] - ref.f999_f1) < 1e-6 and abs(row["f995_f1"] - ref.f995_f1) < 1e-6
    assert row["f999_f1"] - row["f995_f1"] > 0.13  # the one seed that carries the t3 mean
