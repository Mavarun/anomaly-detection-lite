#!/usr/bin/env python3
"""Slice 2: causal streaming detectors, event-level metrics, held-out thresholds,
plus an IsolationForest vs LOF tabular benchmark and a public Nile CUSUM check.

Prints markdown tables used in the README. Offline and seeded (10 seeds).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_lite.benchmark import (  # noqa: E402
    paired_wins,
    run_tabular_benchmark,
    summarise_benchmark,
)
from anomaly_lite.stream_eval import (  # noqa: E402
    nile_cusum_alarms,
    run_streaming_grid,
    summarise_streaming,
)

SEEDS = range(10)


def _pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main() -> None:
    long = run_streaming_grid(seeds=SEEDS)
    print(f"Test-window events per seed: {long['n_test_events'].mean():.1f} "
          f"(min {long['n_test_events'].min()}, max {long['n_test_events'].max()})")
    cols = ["event_f1", "event_precision", "event_recall", "pa_f1", "point_f1",
            "point_far", "mean_delay", "point_roc_auc"]
    s = summarise_streaming(long, cols)
    print("\n### Streaming detectors on the test window (mean ± std, 10 seeds; tolerance 20, "
          "target FAR 0.5%)\n")
    print("| detector | protocol | event F1 | event P | event R | PA-F1 | point F1 | point FAR | "
          "delay (steps) | point ROC-AUC |")
    print("| --- | --- |" + " ---: |" * 8)
    for _, r in s.iterrows():
        cells = [_pm(r[f"{c}_mean"], r[f"{c}_std"], 1 if c == "mean_delay" else 3) for c in cols]
        print(f"| {r.detector} | {r.protocol} | " + " | ".join(cells) + " |")

    t = summarise_streaming(long, ["recall_spike", "recall_level_shift", "recall_variance_burst"])
    t = t[t.protocol != "test_oracle"]
    print("\n### Event recall by injected type (held-out thresholds)\n")
    print("| detector | protocol | spike | level shift | variance burst |")
    print("| --- | --- | ---: | ---: | ---: |")
    for _, r in t.iterrows():
        print(f"| {r.detector} | {r.protocol} | {r.recall_spike_mean:.2f} | "
              f"{r.recall_level_shift_mean:.2f} | {r.recall_variance_burst_mean:.2f} |")

    bench = run_tabular_benchmark(seeds=SEEDS)
    b = summarise_benchmark(bench)
    print("\n### Tabular benchmark: ranking quality (mean ± std, 10 seeds)\n")
    print("| dataset | detector | ROC-AUC | average precision | P@n_anom |")
    print("| --- | --- | ---: | ---: | ---: |")
    for _, r in b.iterrows():
        print(f"| {r.dataset} | {r.detector} | {_pm(r.roc_auc_mean, r.roc_auc_std)} | "
              f"{_pm(r.avg_precision_mean, r.avg_precision_std)} | "
              f"{_pm(r.precision_at_n_anom_mean, r.precision_at_n_anom_std)} |")
    w = paired_wins(bench, "lof_k20", "isolation_forest")
    w2 = paired_wins(bench, "lof_k150", "isolation_forest")
    print("\nSeeds where LOF beats IsolationForest on AP: "
          + "; ".join(f"{d}: k20 {w[d]:.1f}, k150 {w2[d]:.1f}" for d in w.index))

    nile = nile_cusum_alarms()
    print("\n### Nile (statsmodels, 1871-1970): first two-sided CUSUM alarm (k = 0.5)\n")
    print("| reference years | reference ends | h | ref median | ref robust sd | first alarm |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in nile.iterrows():
        alarm = "none" if r.first_alarm_year is None else int(r.first_alarm_year)
        print(f"| {int(r.reference_years)} | {int(r.reference_end)} | {r.h:g} | "
              f"{r.ref_median:.0f} | {r.ref_robust_sd:.1f} | {alarm} |")


if __name__ == "__main__":
    main()
