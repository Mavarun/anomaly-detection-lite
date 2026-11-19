"""Every slice-7 number in the README is recomputed from the committed artifacts."""

import json
from pathlib import Path

import pandas as pd

from anomaly_lite.drift_eval import paired_effects
from anomaly_lite.slice7_eval import decide, posthoc_threshold_reading

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
README = (ROOT / "README.md").read_text()
LONG = pd.read_csv(ART / "slice7_long.csv")


def test_table_rows_match_the_long_grid():
    s = LONG.groupby(["scenario", "detector", "protocol"]).mean(numeric_only=True)
    e = paired_effects(LONG, "protocol", "static_quantile", "static_gpd", "event_f1").set_index(["scenario", "detector"])
    for sc in ("weekend", "weekend_t5", "weekend_t3"):
        for det in ("robust_z", "cusum_k15", "cusum_pw"):
            q, g, r = s.loc[(sc, det, "static_quantile")], s.loc[(sc, det, "static_gpd")], e.loc[(sc, det)]
            row = (f"| {sc} | {det} | {q.event_f1:.3f} / {g.event_f1:.3f} | {r.diff_mean:+.3f} [{r.ci_lo:+.3f}, "
                   f"{r.ci_hi:+.3f}] | {int(r.n_b_better)}/10 | {q.cost_per_1k:.1f} / {g.cost_per_1k:.1f} | "
                   f"{q.threshold:.2f} / {g.threshold:.2f} | {q.realised_far:.4f} / {g.realised_far:.4f} |")
            assert row in README, row


def test_decisions_reproduce_and_match_the_text():
    d = json.loads((ART / "slice7_decisions.json").read_text())
    again = decide(LONG)
    assert d["H1"] is again["H1"] is False and d["H2"] is again["H2"] is False
    assert d["chosen"] == "static_quantile"
    z = d["robust_z_f1_chosen"]
    assert f"{z['weekend']:.3f} on Gaussian vs {z['weekend_t3']:.3f} on t3" in README
    assert f"({z['weekend_t3'] - z['weekend']:+.3f}, budget 0.10)" in README
    assert f"better on only {d['H1_t3']['n_b_better']}/10 seeds" in README


def test_seed_308_and_posthoc_numbers():
    r = LONG[(LONG.seed == 308) & (LONG.detector == "robust_z") & (LONG.scenario == "weekend_t3")].set_index("protocol")
    assert f"jumped to {r.threshold['static_quantile']:.2f} and F1 fell to {r.event_f1['static_quantile']:.3f}" in README
    p = pd.read_csv(ART / "slice7_posthoc.csv")
    for drop in ("none", "308"):
        fresh = posthoc_threshold_reading(LONG, None if drop == "none" else 308).round(4)
        got = p[p.dropped == drop].drop(columns="dropped").reset_index(drop=True)
        pd.testing.assert_frame_equal(got, fresh, check_dtype=False)
    q = p[(p.detector == "robust_z") & (p.scenario == "weekend_t3") & (p.dropped == "none")].set_index("protocol")
    assert f"falls from {q.threshold_sd['static_quantile']:.2f} to\n  {q.threshold_sd['static_gpd']:.2f}" in README
    x = p[(p.detector == "robust_z") & (p.scenario == "weekend_t3") & (p.dropped == "308")].set_index("protocol")
    assert f"{x.event_f1['static_quantile']:.3f} (quantile)\n  vs {x.event_f1['static_gpd']:.3f} (gpd)" in README
