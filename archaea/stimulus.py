"""Poisson input trains (SPEC §3)."""

from __future__ import annotations

import numpy as np

from .neuron import DT_MS


def poisson_spikes_window(
    rng: np.random.Generator,
    rate_hz: float,
    duration_ms: float,
    n_neurons: int,
) -> np.ndarray:
    """
    Independent Poisson spike trains per neuron over [0, duration_ms).

    Returns shape (n_steps, n_neurons) with values in {0.0, 1.0}, step = DT_MS.
    """
    n_steps = int(round(duration_ms / DT_MS))
    if n_steps <= 0:
        raise ValueError("duration too short")
    p = rate_hz * 1e-3 * DT_MS  # rate * dt in seconds for 1ms step
    if p < 0 or p > 1.0:
        raise ValueError(f"invalid spike probability {p} for rate={rate_hz}")
    u = rng.random((n_steps, n_neurons))
    return (u < p).astype(np.float64)


def draw_input_rate(rng: np.random.Generator) -> float:
    """f_in ~ Uniform(10, 100) Hz (SPEC §3.1)."""
    return float(rng.uniform(10.0, 100.0))
