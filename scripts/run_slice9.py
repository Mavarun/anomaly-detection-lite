"""Run slice 9 (pooled fit + calibration static threshold) on the pre-registered seeds."""

from __future__ import annotations

import json
from pathlib import Path

from anomaly_lite.slice9_eval import EVAL_SEEDS, PREREGISTERED, decide, run_slice9_grid

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    df = run_slice9_grid(EVAL_SEEDS)
    df.to_csv(ART / "slice9_pooled.csv", index=False)
    cols = ["quantile_f1", "pooled_f1", "quantile_threshold", "pooled_threshold", "quantile_far", "pooled_far"]
    print(df.groupby(["scenario", "detector"])[cols].mean().round(4).to_string())
    d = decide(df)
    (ART / "slice9_decisions.json").write_text(json.dumps({"preregistered": PREREGISTERED, **d}, indent=2) + "\n")
    print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
