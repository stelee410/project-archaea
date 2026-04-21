"""Unit tests for SPEC v1.2 output-layer synaptic gain g (off-SPEC, opt-in)."""

from __future__ import annotations

import numpy as np
import pytest

from archaea.neuron import N_INPUT, N_WEIGHTS, NetworkBatch, NetworkSingle
from archaea.population import Population
from archaea.stimulus import poisson_spikes_window


def _drive(net: NetworkSingle, rng: np.random.Generator, f_in_hz: float, ms: float) -> int:
    spikes = poisson_spikes_window(rng, f_in_hz, ms, N_INPUT)
    cnt = 0
    for t in range(spikes.shape[0]):
        if net.step(spikes[t]) > 0.0:
            cnt += 1
    return cnt


def test_default_gain_bit_identical():
    """g = 1.0 must be bit-for-bit identical to the no-gain code path."""
    rng_w = np.random.default_rng(1)
    w = rng_w.uniform(-3, 3, size=N_WEIGHTS)
    rng_a = np.random.default_rng(2)
    rng_b = np.random.default_rng(2)
    net_a = NetworkSingle(w, rng=np.random.default_rng(0))
    net_b = NetworkSingle(w, rng=np.random.default_rng(0), output_gain=1.0)
    assert _drive(net_a, rng_a, 80.0, 500.0) == _drive(net_b, rng_b, 80.0, 500.0)


def test_higher_gain_produces_at_least_as_many_spikes_typically_more():
    """Across enough RNG seeds, g=3 should out-fire g=1 on average (often by a lot)."""
    rng_w = np.random.default_rng(3)
    w = rng_w.uniform(-3, 3, size=N_WEIGHTS)
    totals = {1.0: 0, 3.0: 0}
    for seed in range(8):
        for g, total_key in ((1.0, 1.0), (3.0, 3.0)):
            net = NetworkSingle(w, rng=np.random.default_rng(seed), output_gain=g)
            totals[total_key] += _drive(net, np.random.default_rng(100 + seed), 100.0, 500.0)
    assert totals[3.0] >= totals[1.0]
    # we expect a meaningful uplift; allow some headroom for unlucky weights
    assert totals[3.0] > totals[1.0] * 1.05 or totals[1.0] == 0


def test_network_batch_rejects_non_positive_gain():
    rng_w = np.random.default_rng(4)
    W = rng_w.uniform(-1, 1, size=(2, N_WEIGHTS))
    with pytest.raises(ValueError, match="output_gain"):
        NetworkBatch(W, output_gain=0.0)
    with pytest.raises(ValueError, match="output_gain"):
        NetworkBatch(W, output_gain=-1.0)


def test_population_rejects_non_positive_synapse_gain():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="synapse_gain"):
        Population(20, rng, n_initial=20, synapse_gain=0.0)
    rng2 = np.random.default_rng(5)
    with pytest.raises(ValueError, match="synapse_gain"):
        Population(20, rng2, n_initial=20, synapse_gain=-2.0)


def test_population_default_gain_bit_identical():
    """Population with default synapse_gain (==1.0) must match the no-arg path."""
    rng_a = np.random.default_rng(7)
    pop_a = Population(10, rng_a, n_initial=10)
    rng_b = np.random.default_rng(7)
    pop_b = Population(10, rng_b, n_initial=10, synapse_gain=1.0)
    np.testing.assert_array_equal(pop_a.weights, pop_b.weights)
    assert pop_a.synapse_gain == pop_b.synapse_gain == 1.0
