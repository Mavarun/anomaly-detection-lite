"""The slice-5 README section quotes the numbers the harness produces (spot checks on 10 evaluation seeds)."""

from pathlib import Path

import pytest

from anomaly_lite.slice5_eval import decide, run_slice5_grid

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()


@pytest.fixture(scope="module")
def grid():
    return run_slice5_grid(scenarios=("weekly", "weekend"), protocols=("static_quantile",))


def _f1(g, sc, d, det):
    return g[(g.scenario == sc) & (g.deseason == d) & (g.detector == det)].event_f1.mean()


def test_section_order_and_decisions_stated():
    i5, i4 = README.index("## Slice 5 hypotheses"), README.index("## Slice 4 hypotheses")
    assert i5 < README.index("### Weaknesses (slice 5)") < i4
    assert "**H1 passed: ships ON**" in README and "**H2 failed: ships OFF.**" in README


def test_quoted_numbers_match_the_harness(grid):
    for sc, d, det, txt in (("weekend", "harm_dw", "robust_z", "0.738"), ("weekend", "harm_dw3", "robust_z", "0.851"),
                            ("weekend", "harm_dw", "cusum_k15", "0.450"), ("weekend", "harm_dw", "cusum_pw", "0.744"),
                            ("weekly", "harm_dw3", "robust_z", "0.905")):
        assert f"{_f1(grid, sc, d, det):.3f}" == txt and txt in README
    d = decide(grid)
    assert d["H1"] is True and d["H2"] is False and d["recommended_deseason"] == "harm_dw3"
    assert f"{d['H1_weekend']['diff_mean']:+.3f}" == "+0.113" and d["H1_weekend"]["n_b_better"] == 10
