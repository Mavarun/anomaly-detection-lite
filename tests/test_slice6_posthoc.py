"""Post-hoc slice-6 reading (decides nothing): the committed breakdown is reproducible and says what the README says."""

from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_lite.slice6_eval import posthoc_breakdown

ROOT = Path(__file__).resolve().parents[1]


def test_breakdown_recomputes_one_seed_exactly():
    got = posthoc_breakdown(seeds=(3,), scenarios=("weekend_t3",)).sort_values("detector").reset_index(drop=True)
    ref = pd.read_csv(ROOT / "artifacts" / "slice6_posthoc.csv")
    ref = ref[(ref.seed == 3) & (ref.scenario == "weekend_t3")].sort_values("detector").reset_index(drop=True)
    np.testing.assert_allclose(got.threshold, ref.threshold, rtol=1e-5)
    np.testing.assert_array_equal(got.n_false, ref.n_false)


def test_heavy_tails_raise_every_calibrated_threshold_and_clip_kills_spike_recall():
    a = pd.read_csv(ROOT / "artifacts" / "slice6_posthoc.csv").groupby(["scenario", "detector"]).mean(numeric_only=True)
    for det in ("robust_z", "cusum_k15", "cusum_pw", "cusum_pwc"):
        assert a.loc[("weekend_t3", det), "threshold"] > 1.4 * a.loc[("weekend", det), "threshold"]
    # clip=3 caps a spike's push at c - k = 2.5 per step, under the t(3) threshold of ~5.8.
    assert a.loc[("weekend_t3", "cusum_pwc"), "recall_spike"] < 0.15
    assert a.loc[("weekend_t3", "cusum_pw"), "recall_spike"] > 0.6
    # robust_z loses level shifts most.
    drop = a.loc[("weekend", "robust_z")] - a.loc[("weekend_t3", "robust_z")]
    assert drop["recall_level_shift"] > drop["recall_spike"] and drop["recall_level_shift"] > 0.15
