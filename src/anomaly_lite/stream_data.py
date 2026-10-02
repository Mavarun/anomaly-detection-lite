"""Labelled synthetic time series for streaming anomaly-detection research.

Generative model (all seeded, offline):

    x_t = level + A * sin(2 pi t / period) + e_t + injected_t
    e_t = phi * e_{t-1} + sigma * eps_t,  eps_t ~ N(0, 1)

Injected anomaly *events* (contiguous label=1 segments), never overlapping
and separated by ``min_gap`` points, none inside the first ``warmup``
points so an early window can serve as a clean reference:

* ``spike``          -- 1-3 points offset by 5-8 marginal noise sd.
* ``level_shift``    -- 20-60 points offset by 1.5-3 marginal sd.
* ``variance_burst`` -- 30-80 points where the innovation sd is x3-x4.

Spikes are easy for point detectors; small level shifts favour CUSUM-type
accumulation; variance bursts have mean zero and defeat pure mean-shift
detectors -- the mix is deliberate so no single detector wins everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EVENT_TYPES = ("spike", "level_shift", "variance_burst")
_DURATION = {"spike": (1, 3), "level_shift": (20, 60), "variance_burst": (30, 80)}


@dataclass
class StreamData:
    frame: pd.DataFrame  # columns: t, value, label, event_id
    events: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def values(self) -> np.ndarray:
        return self.frame["value"].to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self.frame["label"].to_numpy()


def _place_events(rng, n, n_events, warmup, min_gap, types):
    """Sample non-overlapping (start, end_inclusive, type) triples."""
    events = []
    attempts = 0
    while len(events) < n_events and attempts < 10_000:
        attempts += 1
        kind = types[len(events) % len(types)]
        lo, hi = _DURATION[kind]
        dur = int(rng.integers(lo, hi + 1))
        start = int(rng.integers(warmup, n - dur))
        end = start + dur - 1
        if all(end + min_gap < s or start > e + min_gap for s, e, _ in events):
            events.append((start, end, kind))
    if len(events) < n_events:
        raise ValueError("could not place all events; lower n_events or min_gap")
    return sorted(events)


def make_labelled_stream(
    n: int = 6000,
    period: int = 96,
    season_amp: float = 2.0,
    level: float = 10.0,
    ar_phi: float = 0.6,
    noise_sd: float = 0.5,
    n_events: int = 18,
    warmup: int = 1000,
    min_gap: int = 120,
    event_types: tuple[str, ...] = EVENT_TYPES,
    random_state: int = 0,
) -> StreamData:
    """Generate one labelled stream; see module docstring for the model."""
    if not 0 <= ar_phi < 1:
        raise ValueError("ar_phi must be in [0, 1)")
    unknown = set(event_types) - set(EVENT_TYPES)
    if unknown:
        raise ValueError(f"unknown event types: {sorted(unknown)}")
    rng = np.random.default_rng(random_state)
    marginal_sd = noise_sd / np.sqrt(1.0 - ar_phi**2)
    events = _place_events(rng, n, n_events, warmup, min_gap, tuple(event_types))

    sigma = np.full(n, noise_sd)
    offset = np.zeros(n)
    label = np.zeros(n, dtype=int)
    event_id = np.full(n, -1, dtype=int)
    records = []
    for k, (s, e, kind) in enumerate(events):
        sign = rng.choice([-1.0, 1.0])
        if kind == "spike":
            mag = sign * rng.uniform(5.0, 8.0) * marginal_sd
            offset[s : e + 1] += mag
        elif kind == "level_shift":
            mag = sign * rng.uniform(1.5, 3.0) * marginal_sd
            offset[s : e + 1] += mag
        else:  # variance_burst
            mag = rng.uniform(3.0, 4.0)
            sigma[s : e + 1] *= mag
        label[s : e + 1] = 1
        event_id[s : e + 1] = k
        records.append({"event_id": k, "start": s, "end": e, "type": kind,
                        "magnitude": float(mag), "length": e - s + 1})

    eps = rng.normal(size=n)
    noise = np.empty(n)
    prev = rng.normal(0.0, marginal_sd)
    for t in range(n):
        prev = ar_phi * prev + sigma[t] * eps[t]
        noise[t] = prev
    t_idx = np.arange(n)
    season = season_amp * np.sin(2.0 * np.pi * t_idx / period)
    value = level + season + noise + offset

    frame = pd.DataFrame({"t": t_idx, "value": value, "label": label, "event_id": event_id})
    meta = {"n": n, "period": period, "season_amp": season_amp, "ar_phi": ar_phi,
            "noise_sd": noise_sd, "marginal_sd": float(marginal_sd), "n_events": n_events,
            "warmup": warmup, "prevalence": float(label.mean()), "random_state": random_state}
    return StreamData(frame=frame, events=records, meta=meta)


def contiguous_windows(n: int, fractions: tuple[float, ...] = (0.3, 0.2, 0.5)) -> list[slice]:
    """Split [0, n) into consecutive, non-overlapping time windows.

    Used for fit / threshold-calibration / test windows: no shuffling, so
    nothing from a later window can inform an earlier decision.
    """
    if any(f <= 0 for f in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be positive and sum to 1")
    cuts = np.round(np.cumsum((0.0,) + tuple(fractions)) * n).astype(int)
    cuts[-1] = n
    return [slice(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]
