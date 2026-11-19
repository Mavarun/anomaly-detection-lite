"""Slice 6: heavy-tailed noise and a clipped prewhitened CUSUM (pre-registered decisions).

    python scripts/run_slice6.py            # 10 evaluation seeds, ~15 s on CPU

Writes artifacts/slice6_grid.csv (one row per seed x scenario x detector) and prints the
markdown tables used in the README.
"""

from __future__ import annotations

import json
from pathlib import Path

from anomaly_lite.drift_eval import paired_effects
from anomaly_lite.slice6_eval import DETECTORS6, EVAL_SEEDS, SCENARIOS6, decide, run_slice6_grid

OUT = Path(__file__).resolve().parents[1] / "artifacts"


def main(seeds=EVAL_SEEDS, out: Path = OUT):
    long = run_slice6_grid(seeds=seeds)
    out.mkdir(exist_ok=True)
    long.to_csv(out / "slice6_grid.csv", index=False, float_format="%.6f")
    n = len(set(long.seed))
    agg = long.groupby(["scenario", "detector"]).agg(
        f1=("event_f1", "mean"), f1_sd=("event_f1", "std"), cost=("cost_per_1k", "mean"),
        delay=("mean_delay", "mean"), far=("point_far", "mean")).reset_index()
    print(f"### Event F1 (mean ± std), cost per 1k, delay and point FAR; harm_dw3, static threshold, {n} seeds\n")
    print("| scenario | detector | event F1 | cost/1k | delay | point FAR |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for sc in SCENARIOS6:
        for det in DETECTORS6:
            r = agg[(agg.scenario == sc) & (agg.detector == det)].iloc[0]
            print(f"| {sc} | {det} | {r.f1:.3f} ± {r.f1_sd:.3f} | {r.cost:.1f} | {r.delay:.1f} | {r.far:.4f} |")
    print(f"\n### Paired per-seed effects, cusum_pwc - cusum_pw ({n} seeds; bootstrap 95% CI; exact sign test)\n")
    print("| scenario | metric | mean diff | 95% CI | seeds with diff > 0 | sign p |")
    print("| --- | --- | ---: | :---: | ---: | ---: |")
    for metric in ("event_f1", "cost_per_1k"):
        e = paired_effects(long[long.detector.isin(["cusum_pw", "cusum_pwc"])], "detector", "cusum_pw", "cusum_pwc", metric)
        for _, r in e.iterrows():
            print(f"| {r['scenario']} | {metric} | {r.diff_mean:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | "
                  f"{r.n_b_better}/{r.n_seeds} | {r.sign_p:.3f} |")
    d = decide(long)
    print("\n### Pre-registered decisions\n")
    print(json.dumps({k: d[k] for k in ("H1", "H1_parts", "H2", "robust_z_f1", "recommended_clip")}))
    (out / "slice6_decisions.json").write_text(json.dumps(d, indent=2, default=float) + "\n")
    return d


if __name__ == "__main__":
    main()
