"""Every slice-8 number in the README is recomputed from the committed artifacts."""

import json
from pathlib import Path

import pandas as pd

from anomaly_lite.slice8_eval import decide

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
README = (ROOT / "README.md").read_text()
SEC = README[README.index("## Slice 8 (oracle ceiling)"):README.index("## Slice 7 (peaks-over-threshold")]
DF = pd.read_csv(ART / "slice8_oracle.csv")
FAR = pd.read_csv(ART / "slice8_posthoc_far.csv")


def test_table():
    m = DF.groupby(["scenario", "detector"]).mean(numeric_only=True)
    f = FAR.groupby(["scenario", "detector"]).mean(numeric_only=True)
    for (sc, det), a in m.iterrows():
        b = f.loc[(sc, det)]
        row = (f"| {sc} | {det} | {a.static_f1:.3f} | {a.oracle_f1:.3f} | {b.budget_oracle_f1:.3f} | "
               f"{a.static_threshold:.2f} / {a.oracle_threshold:.2f} | {b.static_far:.4f} / {b.oracle_far:.4f} |")
        assert row in SEC, row


def test_decisions_and_prose():
    d = json.loads((ART / "slice8_decisions.json").read_text())
    again = decide(DF)
    for k in ("H1", "H2_exhausted", "t3_oracle_f1", "gauss_static_f1", "t3_gain"):
        assert d[k] == again[k]
    assert d["H1"] and not d["H2_exhausted"]
    line = d["gauss_static_f1"] - 0.10
    assert f"reaches {d['t3_oracle_f1']:.3f}, above the line of {line:.3f}" in SEC
    assert f"[{d['t3_gain_ci'][0]:.3f}, {d['t3_gain_ci'][1]:.3f}]" in SEC
    r = FAR[(FAR.scenario == "weekend_t3") & (FAR.detector == "robust_z")].mean(numeric_only=True)
    assert f"runs at {r.oracle_far:.2%} FAR, about\n  {r.oracle_far / 0.005:.1f}x" in SEC
    assert f"best t3 F1 is\n  {r.budget_oracle_f1:.3f}" in SEC
    assert f"only by {r.budget_oracle_f1 - line:.3f}" in SEC
    assert f"about\n  {r.budget_oracle_f1 - d['t3_static_f1']:.3f} on the table" in SEC


def test_posthoc_csv_reproduces_for_one_seed():
    from anomaly_lite.slice8_eval import posthoc_oracle_far

    sub = DF[DF.seed == 400].reset_index(drop=True)
    fresh = posthoc_oracle_far(sub)
    old = FAR[FAR.seed == 400].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_dtype=False)


def test_eval_csv_reproduces_for_one_cell():
    from anomaly_lite.slice8_eval import run_slice8_experiment

    fresh = run_slice8_experiment(401, "weekend_t3")
    old = DF[(DF.seed == 401) & (DF.scenario == "weekend_t3")].reset_index(drop=True)
    pd.testing.assert_frame_equal(fresh, old, check_dtype=False)
