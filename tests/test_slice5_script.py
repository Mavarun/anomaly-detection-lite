"""Smoke test: the slice-5 script prints its tables and decisions on two development seeds."""

import importlib.util
from pathlib import Path


def test_script_runs_on_two_dev_seeds(capsys):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_slice5.py"
    spec = importlib.util.spec_from_file_location("run_slice5", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = mod.main(seeds=(100, 101))
    out = capsys.readouterr().out
    for head in ("static FAR-quantile threshold", "Bulk-tail rolling threshold", "Paired per-seed effects",
                 "Pre-registered decisions"):
        assert head in out
    assert out.count("| weekend |") >= 5
    assert d["H1"] is False  # >= 7 of 10 seeds can not be met on two seeds
