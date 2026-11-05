"""Smoke test: the slice-4 script prints its four tables on a tiny grid."""

import importlib.util
from pathlib import Path


def test_script_runs_on_two_seeds(capsys):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_slice4.py"
    spec = importlib.util.spec_from_file_location("run_slice4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(seeds=range(2))
    out = capsys.readouterr().out
    for head in ("by de-seasonaliser", "Threshold protocols", "Paired per-seed effects", "Forgetting factor sensitivity"):
        assert head in out
    assert out.count("| weekly |") >= 7
