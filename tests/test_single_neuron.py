"""Gate A — single output calibration (SPEC §6.1)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from archaea.neuron import DT_MS, DirectLIF10to1
from archaea.stimulus import poisson_spikes_window


def test_gate_a_single_neuron_firing_rate():
    rng = np.random.default_rng(12345)
    w = np.full(10, 1.5, dtype=np.float64)
    cell = DirectLIF10to1(w)
    duration_ms = 1000.0
    n_steps = int(round(duration_ms / DT_MS))
    spikes = poisson_spikes_window(rng, 100.0, duration_ms, 10)
    v_trace = np.zeros(n_steps)
    out_spikes = 0
    for t in range(n_steps):
        v_trace[t] = cell.out.v[0, 0]
        s = cell.step(spikes[t])
        out_spikes += int(s > 0)
    rate_hz = out_spikes / (duration_ms / 1000.0)
    if not (20.0 <= rate_hz <= 120.0):
        Path("diagnostics").mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 3))
        plt.plot(np.arange(n_steps) * DT_MS, v_trace)
        plt.xlabel("t (ms)")
        plt.ylabel("V")
        plt.tight_layout()
        plt.savefig("diagnostics/gate_a_vtrace.png")
        plt.close()
        pytest.fail(f"Gate A failed: output rate {rate_hz:.3f} Hz not in [20, 120]")
