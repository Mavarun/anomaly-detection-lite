"""Every slice-9 number in the README is recomputed from the committed artifacts."""

import json
from pathlib import Path

import pandas as pd

from anomaly_lite.slice9_eval import decide

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
README = (ROOT / "README.md").read_text()
SEC = README[README.index("## Slice 9 (pooled calibration)"):README.index("## Slice 8 (oracle ceiling)")]
DF = pd.read_csv(ART / "slice9_pooled.csv")
P = pd.read_csv(ART / "slice9_posthoc_fit_window.csv")
D = json.loads((ART / "slice9_decisions.json").read_text())


def test_main_table():
    for (sc, det), a in DF.groupby(["scenario", "detector"]).mean(numeric_only=True).iterrows():
        row = (f"| {sc} | {det} | {a.quantile_f1:.3f} / {a.pooled_f1:.3f} | {a.quantile_threshold:.2f} / "
               f"{a.pooled_threshold:.2f} | {a.quantile_far:.4f} / {a.pooled_far:.4f} |")
        assert row in SEC, row


def test_posthoc_table():
    for (sc, det), a in P.groupby(["scenario", "detector"]).mean(numeric_only=True).iterrows():
        assert f"| {sc} | {det} | {a.fit_exceed_rate:.3f} | {a.head_share:.2f} |" in SEC


def test_decisions_and_prose():
    again = decide(DF)
    assert not D["H1"] and not D["H2"] and again["H1"] == D["H1"]
    assert f"changes by {D['t3_f1_diff']:+.3f}, with pooled better on\n  {D['t3_seeds_better']}/10" in SEC
    assert f"it changes by {D['gauss_f1_diff']:+.3f}" in SEC
    assert f"pooled FAR is {D['t3_pooled_far']:.4f}" in SEC
    assert f"t3 F1 {D['t3_f1']:.3f} against a line of\n  {D['gauss_f1'] - 0.10:.3f} (margin {D['guardrail_margin']:+.3f})" in SEC


def test_run_command_listed():
    assert "python scripts/run_slice9.py" in README
