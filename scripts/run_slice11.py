#!/usr/bin/env python3
"""Run slice 11 (forgetting 0.995 vs 0.999) -> artifacts/slice11_forgetting.csv and slice11_decisions.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anomaly_lite import slice11_eval as S  # noqa: E402


def main() -> int:
    df = S.run_slice11_grid()
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    df.round(6).to_csv(out / "slice11_forgetting.csv", index=False)
    dec = S.decide(df)
    (out / "slice11_decisions.json").write_text(json.dumps({"preregistered": S.PREREGISTERED, "decision": dec},
                                                           indent=2) + "\n")
    print(df.groupby("scenario")[["f995_f1", "f999_f1", "f995_far", "f999_far", "f995_threshold",
                                  "f999_threshold"]].mean().round(4).to_string())
    print(json.dumps(dec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
