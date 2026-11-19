"""Run slice 8 (oracle ceiling for static thresholds) on the pre-registered seeds."""

from __future__ import annotations

import json
from pathlib import Path

from anomaly_lite.slice8_eval import EVAL_SEEDS, PREREGISTERED, decide, run_slice8_grid

ART = Path(__file__).resolve().parents[1] / "artifacts"


def main() -> None:
    df = run_slice8_grid(EVAL_SEEDS)
    df.to_csv(ART / "slice8_oracle.csv", index=False)
    summ = df.groupby(["scenario", "detector"])[["static_f1", "oracle_f1", "static_threshold", "oracle_threshold",
                                                 "oracle_quantile"]].mean().round(4)
    d = decide(df)
    (ART / "slice8_decisions.json").write_text(json.dumps({"preregistered": PREREGISTERED, **d}, indent=2) + "\n")
    print(summ.to_string())
    print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
