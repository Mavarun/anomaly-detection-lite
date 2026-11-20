"""Slice 9 committed artifacts: decisions recompute from the csv, and stored seeds reproduce."""

import json
from pathlib import Path

import pandas as pd
import pytest

from anomaly_lite import slice9_eval as S

ART = Path(__file__).resolve().parents[1] / "artifacts"
DF = pd.read_csv(ART / "slice9_pooled.csv")
DEC = json.loads((ART / "slice9_decisions.json").read_text())


def test_grid_and_decisions_recompute():
    assert sorted(DF.seed.unique()) == list(S.EVAL_SEEDS) and len(DF) == 90
    d = S.decide(DF)
    for k, v in d.items():
        assert DEC[k] == (pytest.approx(v) if isinstance(v, float) else v), k
    assert DEC["preregistered"] == S.PREREGISTERED


def test_outcome_h1_fails_on_f1_not_far_and_h2_fails():
    assert not DEC["H1"] and DEC["rule"] == "quantile"
    assert DEC["t3_f1_diff"] < 0 and DEC["t3_seeds_better"] < 7 and DEC["gauss_f1_diff"] < -0.01
    assert DEC["t3_pooled_far"] <= S.FAR_CAP  # the cap the expectation predicted would bind did not
    assert not DEC["H2"] and DEC["guardrail_margin"] < 0


def test_pooled_thresholds_are_higher_everywhere():
    """The expectation (in-sample scores too tight) was wrong in sign: pooling raises every mean threshold."""
    m = DF.groupby(["scenario", "detector"])[["quantile_threshold", "pooled_threshold"]].mean()
    assert (m.pooled_threshold > m.quantile_threshold).all()


@pytest.mark.parametrize("seed,scenario", [(500, "weekend_t3"), (509, "weekend")])
def test_stored_rows_reproduce(seed, scenario):
    new = S.run_slice9_experiment(seed, scenario).set_index("detector")
    old = DF[(DF.seed == seed) & (DF.scenario == scenario)].set_index("detector")
    for c in ("quantile_threshold", "pooled_threshold", "quantile_f1", "pooled_f1", "pooled_far"):
        assert new[c].to_numpy() == pytest.approx(old.loc[new.index, c].to_numpy(), rel=1e-9, abs=1e-12)
