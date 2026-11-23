"""Every slice-10 number in the README comes from the committed artifacts."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SEC = README[README.index("## Slice 10 (pooled after"): README.index("## Slice 9 (pooled calibration)")]
DF = pd.read_csv(ROOT / "artifacts" / "slice10_pooled_weekly.csv")
D = json.loads((ROOT / "artifacts" / "slice10_decisions.json").read_text())


def test_table():
    m = DF.groupby(["scenario", "detector"]).mean(numeric_only=True)
    for (sc, det), r in m.iterrows():
        row = (f"| {sc} | {det} | {r.fit_exceed_rate:.4f} | {r.quantile_f1:.3f} / {r.pooled_weekly_f1:.3f} | "
               f"{r.quantile_threshold:.2f} / {r.pooled_weekly_threshold:.2f} | {r.quantile_far:.4f} / {r.pooled_weekly_far:.4f} |")
        assert row in SEC, row


def test_prose():
    assert f"falls to {D['C0_exceed']['weekend']:.4f} (Gaussian) and {D['C0_exceed']['weekend_t3']:.4f} (t3)" in SEC
    assert f"F1 changes by {D['t3_f1_diff']:+.3f}" in SEC and f"{D['t3_seeds_better']}/10 seeds" in SEC
    assert f"gains {D['gauss_f1_diff']:+.3f} ({D['gauss_seeds_better']}/10)" in SEC
    assert f"({D['t3_pooled_weekly_far']:.4f})" in SEC
    assert f"margin {D['guardrail_margin']:+.3f}" in SEC and not D["H2"]
    rz = DF[(DF.detector == "robust_z") & (DF.scenario == "weekend_t3")].mean(numeric_only=True)
    assert f"({rz.quantile_threshold:.2f} -> {rz.pooled_weekly_threshold:.2f})" in SEC
    cus = DF[DF.detector.str.startswith("cusum")].groupby(["scenario", "detector"]).fit_exceed_rate.mean()
    assert "(fit-window exceed rates 4-9%)" in SEC and cus.min() >= 0.035 and cus.max() < 0.095
    # slice 9's numbers this section compares against
    assert "0.066 and 0.035 in slice 9" in SEC and "| weekend | robust_z | 0.066 |" in README
    assert "| weekend_t3 | robust_z | 0.035 |" in README
