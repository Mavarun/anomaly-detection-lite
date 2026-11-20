"""Slice 9 post-hoc: fit-window scores are not a stationary sample of the held-out score."""

from pathlib import Path

import pandas as pd
import pytest

from anomaly_lite import slice9_eval as S

P = pd.read_csv(Path(__file__).resolve().parents[1] / "artifacts" / "slice9_posthoc_fit_window.csv")


def test_fit_window_exceeds_the_held_out_threshold_far_above_nominal():
    m = P.groupby(["scenario", "detector"]).fit_exceed_rate.mean()
    assert len(P) == 60 and (m > 5 * S.RATE).all()
    assert m.loc[("weekend_t3", "cusum_k15")] > 0.3


def test_exceedances_crowd_the_head_of_the_fit_window_for_robust_z():
    m = P.groupby(["scenario", "detector"]).head_share.mean()
    even = P.head_share_if_even.iloc[0]
    assert even == pytest.approx(1 / 3)
    assert (m.xs("robust_z", level="detector") > even).all()


def test_stored_seed_reproduces():
    new = S.posthoc_fit_window(seeds=(503,), scenarios=("weekend_t3",)).set_index("detector")
    old = P[(P.seed == 503) & (P.scenario == "weekend_t3")].set_index("detector")
    assert new.fit_exceed_rate.to_numpy() == pytest.approx(old.loc[new.index].fit_exceed_rate.to_numpy())
