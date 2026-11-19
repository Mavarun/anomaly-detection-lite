"""Run slice 7 (GPD threshold) on the pre-registered evaluation seeds and write the artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from anomaly_lite.drift_eval import paired_effects
from anomaly_lite.slice7_eval import EVAL_SEEDS, PREREGISTERED, decide, run_slice7_grid

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    long = run_slice7_grid(EVAL_SEEDS)
    long.to_csv(ART / "slice7_long.csv", index=False)
    summ = (long.groupby(["scenario", "detector", "protocol"])
            [["event_f1", "event_recall", "event_precision", "cost_per_1k", "threshold", "realised_far"]]
            .mean().round(4).reset_index())
    summ.to_csv(ART / "slice7_summary.csv", index=False)
    eff = paired_effects(long, "protocol", "static_quantile", "static_gpd", "event_f1")
    eff.to_csv(ART / "slice7_effects.csv", index=False)
    d = decide(long)
    (ART / "slice7_decisions.json").write_text(json.dumps({"preregistered": PREREGISTERED, **d}, indent=2,
                                                          default=float) + "\n")
    print(summ.to_string(index=False))
    print(eff.round(4).to_string(index=False))
    print(json.dumps(d, indent=2, default=float))


if __name__ == "__main__":
    main()
