#!/usr/bin/env python3
"""Slice 4: second seasonality, harmonic RLS de-seasonaliser, freezable CUSUM, bulk-tail rolling thresholds.
Prints the README tables. Offline, 10 evaluation seeds (0-9), ~2 min on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from anomaly_lite.slice4_eval import DETECTORS4, paired_effects, run_slice4_grid, summarise  # noqa: E402

SEEDS = range(10)


def _pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def _pairs(long, scen, deseason, protocol, detectors, factor, a, b, metric="event_f1"):
    sub = long[long.scenario.isin(scen) & long.deseason.isin(deseason) & long.protocol.isin(protocol)
               & long.detector.isin(detectors)]
    return paired_effects(sub, factor, a, b, metric)


def main(seeds=SEEDS) -> None:
    long = run_slice4_grid(seeds=seeds)
    s = summarise(long, ["event_f1", "point_far", "cost_per_1k"])
    sq = s[s.protocol == "static_quantile"]

    print("### Event F1 / point FAR / cost per 1k by de-seasonaliser, held-out FAR-quantile threshold "
          "(test window, mean ± std, 10 seeds)\n")
    print("| scenario | detector | " + " | ".join(f"{d} F1" for d in ("static", "online", "harm_d", "harm_dw")) + " | "
          + " | ".join(f"{d} FAR" for d in ("online", "harm_dw")) + " | cost/1k online | cost/1k harm_dw |")
    print("| --- | --- |" + " ---: |" * 8)
    for sc in ("drift", "weekly", "weekly_missing"):
        for det in DETECTORS4:
            r = {d: sq[(sq.scenario == sc) & (sq.deseason == d) & (sq.detector == det)].iloc[0]
                 for d in ("static", "online", "harm_d", "harm_dw")}
            print(f"| {sc} | {det} | " + " | ".join(_pm(r[d].event_f1_mean, r[d].event_f1_std) for d in r) + " | "
                  f"{r['online'].point_far_mean:.3f} | {r['harm_dw'].point_far_mean:.3f} | "
                  f"{r['online'].cost_per_1k_mean:.1f} | {r['harm_dw'].cost_per_1k_mean:.1f} |")

    print("\n### Threshold protocols, weekly scenario, harm_dw residuals (mean over 10 seeds)\n")
    print("| detector | protocol | event F1 | point FAR | cost / 1k |")
    print("| --- | --- | ---: | ---: | ---: |")
    sw = s[(s.scenario == "weekly") & (s.deseason == "harm_dw")]
    for det in DETECTORS4:
        for p in ("static_quantile", "rolling_quantile", "rolling_rescaled", "rolling_bulk"):
            r = sw[(sw.detector == det) & (sw.protocol == p)].iloc[0]
            print(f"| {det} | {p} | {r.event_f1_mean:.3f} | {r.point_far_mean:.4f} | {r.cost_per_1k_mean:.1f} |")

    print("\n### Paired per-seed effects (10 seeds; bootstrap 95% CI; exact sign test)\n")
    print("| comparison (b - a) | scenario | detector | metric | mean diff | 95% CI | seeds with diff > 0 | sign p |")
    print("| --- | --- | --- | --- | ---: | :---: | ---: | ---: |")
    rows = []
    for metric in ("event_f1", "cost_per_1k"):
        e = _pairs(long, ("weekly", "drift"), ("online", "harm_dw"), ("static_quantile",), ("robust_z", "ewma"),
                   "deseason", "online", "harm_dw", metric)
        rows += [("harm_dw - online", r) for _, r in e.iterrows()]
    e = _pairs(long, ("weekly", "drift"), ("harm_d", "harm_dw"), ("static_quantile",), ("robust_z",),
               "deseason", "harm_d", "harm_dw")
    rows += [("harm_dw - harm_d", r) for _, r in e.iterrows()]
    e = _pairs(long, ("weekly",), ("harm_dw",), ("static_quantile", "rolling_bulk"), ("robust_z", "ewma"),
               "protocol", "static_quantile", "rolling_bulk")
    rows += [("rolling_bulk - static thr", r) for _, r in e.iterrows()]
    e = _pairs(long, ("weekly",), ("harm_dw",), ("static_quantile", "rolling_bulk"), ("cusum_k15", "variance"),
               "protocol", "static_quantile", "rolling_bulk")
    rows += [("rolling_bulk - static thr", r) for _, r in e.iterrows()]
    e = _pairs(long, ("weekly",), ("harm_dw",), ("rolling_quantile", "rolling_bulk"), ("robust_z", "ewma"),
               "protocol", "rolling_quantile", "rolling_bulk")
    rows += [("rolling_bulk - plain rolling rank", r) for _, r in e.iterrows()]
    e = _pairs(long, ("weekly",), ("harm_dw",), ("rolling_rescaled", "rolling_bulk"), ("robust_z", "ewma"),
               "protocol", "rolling_rescaled", "rolling_bulk")
    rows += [("rolling_bulk - rescaled", r) for _, r in e.iterrows()]
    cs = long[(long.scenario.isin(["weekly", "drift"])) & (long.deseason == "harm_dw")
              & (long.protocol == "static_quantile")]
    for a, b, name in (("cusum", "cusum_k15", "CUSUM k 1.5 - k 0.5"),
                       ("cusum_k15", "cusum_frozen_k15", "frozen - fixed ref (k 1.5)"),
                       ("cusum", "cusum_frozen", "frozen - fixed ref (k 0.5)"),
                       ("robust_z", "cusum_k15", "CUSUM k 1.5 - robust z")):
        e = paired_effects(cs[cs.detector.isin([a, b])], "detector", a, b, "event_f1")
        rows += [(name, r) for _, r in e.iterrows()]
    for name, r in rows:
        det = r["detector"] if "detector" in r.index else "see comparison"
        print(f"| {name} | {r['scenario']} | {det} | {r.metric} | {r.diff_mean:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | "
              f"{r.n_b_better}/{r.n_seeds} | {r.sign_p:.3f} |")

    print("\n### Forgetting factor sensitivity (harm_dw, weekly, static threshold, robust z / cusum_k15 event F1)\n")
    print("| forgetting | memory (steps) | robust_z F1 | robust_z cost/1k | cusum_k15 F1 |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for fg in (0.98, 0.99, 0.995, 0.999):
        g = run_slice4_grid(seeds=seeds, scenarios=("weekly",), deseason=("harm_dw",), protocols=("static_quantile",),
                            forgetting=fg)
        m = g.groupby("detector")[["event_f1", "cost_per_1k"]].mean()
        print(f"| {fg} | {1 / (1 - fg):.0f} | {m.loc['robust_z', 'event_f1']:.3f} | {m.loc['robust_z', 'cost_per_1k']:.1f} | "
              f"{m.loc['cusum_k15', 'event_f1']:.3f} |")


if __name__ == "__main__":
    main()
