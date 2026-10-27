#!/usr/bin/env python3
"""Slice 3: drift, missing data, an online de-seasonaliser, a variance detector and
walk-forward thresholds, with alarm costs. Prints the README tables. Offline, 10 seeds (~30 s).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_lite.drift_eval import paired_effects, run_drift_grid, summarise  # noqa: E402

SEEDS = range(10)
DETS = ("ewma", "cusum", "robust_z", "variance", "rz_or_var", "random")


def _pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main() -> None:
    long = run_drift_grid(seeds=SEEDS)
    s = summarise(long, ["event_f1", "point_far", "cost_per_1k", "mean_delay"])

    print("### Held-out FAR-quantile threshold (static_quantile): fit-once vs online de-seasonalising "
          "(test window, mean ± std, 10 seeds)\n")
    print("| scenario | detector | event F1, fit-once | event F1, online | point FAR, fit-once | "
          "point FAR, online | cost / 1k, fit-once | cost / 1k, online |")
    print("| --- | --- |" + " ---: |" * 6)
    sq = s[s.protocol == "static_quantile"]
    for sc in ("stationary", "drift", "drift_missing"):
        for d in DETS:
            a = sq[(sq.scenario == sc) & (sq.deseason == "static") & (sq.detector == d)].iloc[0]
            b = sq[(sq.scenario == sc) & (sq.deseason == "online") & (sq.detector == d)].iloc[0]
            print(f"| {sc} | {d} | {_pm(a.event_f1_mean, a.event_f1_std)} | {_pm(b.event_f1_mean, b.event_f1_std)} | "
                  f"{a.point_far_mean:.3f} | {b.point_far_mean:.3f} | {_pm(a.cost_per_1k_mean, a.cost_per_1k_std, 1)} | "
                  f"{_pm(b.cost_per_1k_mean, b.cost_per_1k_std, 1)} |")

    print("\n### Threshold protocols on the drift scenario, online de-seasonalising (mean over 10 seeds)\n")
    print("| detector | protocol | event F1 | point FAR | cost / 1k |")
    print("| --- | --- | ---: | ---: | ---: |")
    sd = s[(s.scenario == "drift") & (s.deseason == "online")]
    for d in ("ewma", "cusum", "robust_z", "variance", "random"):
        for p in ("static_quantile", "rolling_quantile", "rolling_rescaled"):
            r = sd[(sd.detector == d) & (sd.protocol == p)].iloc[0]
            print(f"| {d} | {p} | {r.event_f1_mean:.3f} | {r.point_far_mean:.4f} | {r.cost_per_1k_mean:.1f} |")

    print("\n### Paired per-seed effects (10 seeds; bootstrap 95% CI; exact sign test)\n")
    print("| comparison (b - a) | scenario | detector | metric | mean diff | 95% CI | seeds with diff > 0 | sign p |")
    print("| --- | --- | --- | --- | ---: | :---: | ---: | ---: |")
    sqo = long[long.protocol == "static_quantile"]
    rows = []
    for metric in ("event_f1", "cost_per_1k"):
        e = paired_effects(sqo, "deseason", "static", "online", metric)
        rows += [("online - fit-once", r) for _, r in e.iterrows()
                 if r.detector in ("ewma", "cusum", "robust_z") and r.scenario in ("stationary", "drift")]
    on = long[long.deseason == "online"]
    on = on[on.protocol.isin(["static_quantile", "rolling_rescaled"])]
    e = paired_effects(on, "protocol", "static_quantile", "rolling_rescaled", "event_f1")
    rows += [("rescaled - static thr", r) for _, r in e.iterrows()
             if r.detector in ("ewma", "robust_z") and r.scenario == "drift"]
    pr = long[(long.deseason == "online") & (long.protocol == "static_quantile")
              & long.scenario.isin(["drift", "drift_missing"])]
    e = paired_effects(pr, "scenario", "drift", "drift_missing", "event_f1")
    rows += [("5% missing - none", r) for _, r in e.iterrows() if r.detector in ("ewma", "robust_z")]
    zz = long[(long.deseason == "online") & (long.protocol == "static_quantile")
              & long.detector.isin(["robust_z", "rz_or_var"]) & (long.scenario == "drift")]
    e = paired_effects(zz, "detector", "robust_z", "rz_or_var", "event_f1")
    rows += [("rz_or_var - robust_z", r) for _, r in e.iterrows()]
    for name, r in rows:
        sc = r["scenario"] if "scenario" in r.index else "drift vs drift_missing"
        det = r["detector"] if "detector" in r.index else "robust_z vs rz_or_var"
        print(f"| {name} | {sc} | {det} | {r.metric} | {r.diff_mean:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | "
              f"{r.n_b_better}/{r.n_seeds} | {r.sign_p:.3f} |")


if __name__ == "__main__":
    main()
