"""Numbers quoted in the README's slice 11 section match the committed artifacts."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
SEC = README[README.index("## Slice 11 (longer memory"): README.index("## Slice 10 (pooled after")]
DF = pd.read_csv(ROOT / "artifacts" / "slice11_forgetting.csv")
DEC = json.loads((ROOT / "artifacts" / "slice11_decisions.json").read_text())["decision"]


def test_newest_first_and_run_line():
    assert README.index("## Slice 11 (longer memory") < README.index("## Slice 10 (pooled after")
    assert "python scripts/run_slice11.py" in README


def test_table():
    m = DF.groupby("scenario").mean(numeric_only=True)
    for sc in ("weekend", "weekend_t3", "weekend_t5"):
        r = m.loc[sc]
        x = DF[DF.scenario == sc]
        better = int(((x.f999_f1 - x.f995_f1) > 0).sum())
        line = (f"| {sc} | {r.f995_f1:.3f} / {r.f999_f1:.3f} | {better}/10 | {r.f995_far:.4f} / {r.f999_far:.4f} | "
                f"{r.f995_threshold:.2f} / {r.f999_threshold:.2f} |")
        assert line in SEC, line


def test_claims():
    assert f"(+{DEC['t3_f1_diff']:.3f})" in SEC and f"Gaussian +{DEC['gauss_f1_diff']:.3f}" in SEC
    assert f"margin {DEC['guardrail_margin']:.3f}" in SEC and f"t3 FAR {DEC['t3_f999_far']:.4f}" in SEC
    x = DF[(DF.scenario == "weekend_t3") & (DF.seed == 705)].iloc[0]
    assert f"705: +{x.f999_f1 - x.f995_f1:.3f}" in SEC
