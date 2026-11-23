"""Run slice 10 (pooled calibration after the weekly de-seasonaliser) on the pre-registered seeds."""

from __future__ import annotations

import json
from pathlib import Path

from anomaly_lite.slice10_eval import EVAL_SEEDS, PREREGISTERED, decide, run_slice10_grid

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    df = run_slice10_grid(EVAL_SEEDS)
    df.to_csv(ART / "slice10_pooled_weekly.csv", index=False)
    cols = ["fit_exceed_rate", "quantile_f1", "pooled_weekly_f1", "quantile_threshold", "pooled_weekly_threshold",
            "quantile_far", "pooled_weekly_far"]
    print(df.groupby(["scenario", "detector"])[cols].mean().round(4).to_string())
    d = decide(df)
    (ART / "slice10_decisions.json").write_text(json.dumps({"preregistered": PREREGISTERED, **d}, indent=2) + "\n")
    print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
