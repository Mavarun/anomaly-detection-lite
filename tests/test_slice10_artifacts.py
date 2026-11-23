import json
from pathlib import Path

import pandas as pd

from anomaly_lite import slice10_eval as S

ART = Path(__file__).resolve().parents[1] / "artifacts"
DF = pd.read_csv(ART / "slice10_pooled_weekly.csv")
DEC = json.loads((ART / "slice10_decisions.json").read_text())


def test_shape():
    assert len(DF) == 3 * 10 * 3 and sorted(DF.seed.unique()) == list(S.EVAL_SEEDS)


def test_decisions_recompute():
    d = S.decide(DF)
    for k in ("C0", "H1", "H2", "rule", "t3_seeds_better"):
        assert d[k] == DEC[k]
    assert d["C0"] and not d["H1"] and not d["H2"] and d["rule"] == "quantile"
    assert d["t3_seeds_better"] == 4 and d["t3_f1_diff"] < 0 < d["gauss_f1_diff"]


def test_cusums_stay_wide_and_pooling_raises_their_thresholds():
    m = DF.groupby(["scenario", "detector"]).mean(numeric_only=True)
    for sc in ("weekend", "weekend_t3", "weekend_t5"):
        for det in ("cusum_k15", "cusum_pw"):
            assert m.loc[(sc, det)].fit_exceed_rate > 0.03
            assert m.loc[(sc, det)].pooled_weekly_threshold > 1.5 * m.loc[(sc, det)].quantile_threshold


def test_one_seed_reproduces():
    fresh = S.run_slice10_experiment(600, "weekend_t3")
    old = DF[(DF.seed == 600) & (DF.scenario == "weekend_t3")].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_exact=False, atol=1e-9)
