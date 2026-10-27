"""Smoke test: the slice-3 script prints every README table (1 seed)."""

import importlib.util
from pathlib import Path


def test_script_prints_all_tables(capsys, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_drift_slice.py"
    spec = importlib.util.spec_from_file_location("run_drift_slice", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SEEDS", range(2))
    mod.main()
    out = capsys.readouterr().out
    assert "fit-once vs online" in out and "Threshold protocols" in out and "Paired per-seed effects" in out
    assert out.count("| drift | robust_z |") >= 1 and "| drift_missing | random |" in out
    assert "rolling_rescaled" in out and "sign p" in out
