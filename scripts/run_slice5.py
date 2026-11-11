#!/usr/bin/env python3
"""Slice 5: weekend-dip weekly shape, per-period harmonics, prewhitened CUSUM.
Prints the README tables and the pre-registered H1/H2 decisions. Offline, evaluation seeds 0-9.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_lite.slice5_eval import (  # noqa: E402
    DETECTORS5, EVAL_SEEDS, decide, paired_effects, run_slice5_grid, summarise,
)


def _pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main(seeds=EVAL_SEEDS) -> dict:
    long = run_slice5_grid(seeds=seeds)
    s = summarise(long, ["event_f1", "point_far", "cost_per_1k", "mean_delay"])
    n = len(tuple(seeds))

    print(f"### Event F1 (mean ± std) and cost per 1k, static FAR-quantile threshold, test window, {n} seeds\n")
    print("| scenario | detector | harm_dw F1 | harm_dw3 F1 | harm_dw cost/1k | harm_dw3 cost/1k | harm_dw3 delay |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    sq = s[s.protocol == "static_quantile"]
    for sc in ("weekly", "weekend", "weekend_missing"):
        for det in DETECTORS5:
            r = {d: sq[(sq.scenario == sc) & (sq.deseason == d) & (sq.detector == det)].iloc[0]
                 for d in ("harm_dw", "harm_dw3")}
            print(f"| {sc} | {det} | {_pm(r['harm_dw'].event_f1_mean, r['harm_dw'].event_f1_std)} | "
                  f"{_pm(r['harm_dw3'].event_f1_mean, r['harm_dw3'].event_f1_std)} | {r['harm_dw'].cost_per_1k_mean:.1f} | "
                  f"{r['harm_dw3'].cost_per_1k_mean:.1f} | {r['harm_dw3'].mean_delay_mean:.1f} |")

    print(f"\n### Bulk-tail rolling threshold (context only), harm_dw3, event F1 / cost per 1k\n")
    print("| scenario | " + " | ".join(DETECTORS5) + " |")
    print("| --- |" + " ---: |" * len(DETECTORS5))
    rb = s[(s.protocol == "rolling_bulk") & (s.deseason == "harm_dw3")]
    for sc in ("weekly", "weekend", "weekend_missing"):
        cells = []
        for det in DETECTORS5:
            r = rb[(rb.scenario == sc) & (rb.detector == det)].iloc[0]
            cells.append(f"{r.event_f1_mean:.3f} / {r.cost_per_1k_mean:.1f}")
        print(f"| {sc} | " + " | ".join(cells) + " |")

    print(f"\n### Paired per-seed effects ({n} seeds; bootstrap 95% CI; exact sign test), static threshold\n")
    print("| comparison (b - a) | scenario | slice | metric | mean diff | 95% CI | seeds with diff > 0 | sign p |")
    print("| --- | --- | --- | --- | ---: | :---: | ---: | ---: |")
    st = long[long.protocol == "static_quantile"]
    rows = []
    for det in ("robust_z", "ewma", "cusum_pw"):
        e = paired_effects(st[st.detector == det], "deseason", "harm_dw", "harm_dw3", "event_f1")
        rows += [("harm_dw3 - harm_dw", r, det) for _, r in e.iterrows()]
    for dsn in ("harm_dw", "harm_dw3"):
        sub = st[st.deseason == dsn]
        for a, b, name in (("cusum_k15", "cusum_pw", "cusum_pw - cusum_k15"), ("cusum", "cusum_pw", "cusum_pw - raw k 0.5"),
                           ("robust_z", "cusum_pw", "cusum_pw - robust_z")):
            for metric in ("event_f1", "cost_per_1k"):
                e = paired_effects(sub[sub.detector.isin([a, b])], "detector", a, b, metric)
                rows += [(name, r, dsn) for _, r in e.iterrows()]
    for name, r, sl in rows:
        print(f"| {name} | {r['scenario']} | {sl} | {r.metric} | {r.diff_mean:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | "
              f"{r.n_b_better}/{r.n_seeds} | {r.sign_p:.3f} |")

    d = decide(long)
    print("\n### Pre-registered decisions\n")
    print(json.dumps({k: d[k] for k in ("H1", "recommended_deseason", "H2", "H2_f1_nonneg", "H2_wins6", "H2_cost_ok")}))
    print(f"H1 weekend robust_z diff {d['H1_weekend']['diff_mean']:+.3f} ({d['H1_weekend']['n_b_better']}/10), "
          f"weekly diff {d['H1_weekly']['diff_mean']:+.3f}")
    return d


if __name__ == "__main__":
    main()
