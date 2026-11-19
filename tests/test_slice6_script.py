"""The slice-6 script runs on dev seeds into a temp dir, and the committed artifact is consistent."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_script_runs_on_two_dev_seeds(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("run_slice6", ROOT / "scripts" / "run_slice6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.main(seeds=(100, 101), out=tmp_path)
    out = capsys.readouterr().out
    for head in ("Event F1 (mean ± std)", "Paired per-seed effects", "Pre-registered decisions"):
        assert head in out
    assert (tmp_path / "slice6_grid.csv").exists()
    assert d["H1"] is False  # >= 7 of 10 seeds can not be met on two seeds


def test_committed_grid_covers_eval_seeds_and_reproduces_slice5_control():
    g = pd.read_csv(ROOT / "artifacts" / "slice6_grid.csv")
    assert sorted(set(g.seed)) == list(range(10)) and len(g) == 10 * 3 * 4
    f1 = g.groupby(["scenario", "detector"]).event_f1.mean()
    # The Gaussian control is the slice-5 'weekend' stream with harm_dw3: same numbers as the slice-5 table.
    assert round(f1[("weekend", "robust_z")], 3) == 0.851
    assert round(f1[("weekend", "cusum_k15")], 3) == 0.709
    assert round(f1[("weekend", "cusum_pw")], 3) == 0.759


def test_committed_decisions_match_a_recompute_from_the_grid():
    from anomaly_lite.slice6_eval import decide

    g = pd.read_csv(ROOT / "artifacts" / "slice6_grid.csv")
    d = decide(g)
    rec = json.loads((ROOT / "artifacts" / "slice6_decisions.json").read_text())
    assert d["H1"] == rec["H1"] is False and d["H2"] == rec["H2"] is False
    assert d["H1_parts"] == rec["H1_parts"] == {"t3_gain": False, "gauss_ok": True, "cost_ok": False}
    assert rec["recommended_clip"] is None
