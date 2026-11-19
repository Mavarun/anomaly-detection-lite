"""Every slice-6 number in the README is recomputed from the committed artifacts."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def test_f1_and_cost_table_matches_grid():
    g = pd.read_csv(ROOT / "artifacts" / "slice6_grid.csv").groupby(["scenario", "detector"])
    f1, sd, cost = g.event_f1.mean(), g.event_f1.std(), g.cost_per_1k.mean()
    for sc, label in (("weekend", "weekend (Gaussian)"), ("weekend_t5", "weekend_t5"), ("weekend_t3", "weekend_t3")):
        cells = [f"{f1[(sc, d)]:.3f} ± {sd[(sc, d)]:.3f} / {cost[(sc, d)]:.1f}"
                 for d in ("robust_z", "cusum_k15", "cusum_pw", "cusum_pwc")]
        assert f"| {label} | " + " | ".join(cells) + " |" in README


def test_decisions_and_guardrail_numbers():
    d = json.loads((ROOT / "artifacts" / "slice6_decisions.json").read_text())
    assert d["H1"] is False and d["H2"] is False
    z = d["robust_z_f1"]
    assert f"{z['weekend']:.3f} to {z['weekend_t3']:.3f}" in README
    assert f"({z['weekend_t3'] - z['weekend']:+.3f}, budget 0.10)" in README


def test_clip_paired_table_matches_the_script_bootstrap():
    # Same call as scripts/run_slice6.py (all scenarios in one paired_effects call, seed 0).
    from anomaly_lite.drift_eval import paired_effects

    g = pd.read_csv(ROOT / "artifacts" / "slice6_grid.csv")
    pair = g[g.detector.isin(["cusum_pw", "cusum_pwc"])]
    f1 = paired_effects(pair, "detector", "cusum_pw", "cusum_pwc", "event_f1").set_index("scenario")
    cost = paired_effects(pair, "detector", "cusum_pw", "cusum_pwc", "cost_per_1k").set_index("scenario")
    for sc in ("weekend", "weekend_t5", "weekend_t3"):
        r = f1.loc[sc]
        assert (f"| {sc} | {r.diff_mean:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.n_b_better}/10 | "
                f"{cost.loc[sc].diff_mean:+.1f} |") in README


def test_posthoc_numbers():
    a = pd.read_csv(ROOT / "artifacts" / "slice6_posthoc.csv").groupby(["scenario", "detector"]).mean(numeric_only=True)
    r = lambda sc, det, c, nd=3: f"{a.loc[(sc, det), c]:.{nd}f}"
    assert f"robust z {r('weekend', 'robust_z', 'threshold', 2)} -> {r('weekend_t3', 'robust_z', 'threshold', 2)}" in README
    assert f"from {r('weekend', 'cusum_pwc', 'recall_spike')} to {r('weekend_t3', 'cusum_pwc', 'recall_spike')}" in README
    assert f"(recall {r('weekend', 'robust_z', 'recall_level_shift')} -> {r('weekend_t3', 'robust_z', 'recall_level_shift')})" in README
