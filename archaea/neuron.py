"""LIF neurons and 10→20→1 feedforward network (SPEC §2, §1.1)."""

from __future__ import annotations

import numpy as np

# Mandatory constants (SPEC §2.2)
DT_MS = 1.0
TAU_MS = 20.0
R_MEM = 1.0
V_REST = 0.0
V_THRESHOLD = 1.0
V_RESET = 0.0
T_REFRACTORY_MS = 2.0
I_IN = 2.5

N_INPUT = 10
N_HIDDEN = 20
N_OUTPUT = 1
N_WEIGHTS = N_INPUT * N_HIDDEN + N_HIDDEN * N_OUTPUT  # 220

REFRACTORY_STEPS = int(T_REFRACTORY_MS / DT_MS)  # 2
DT_OVER_TAU = DT_MS / TAU_MS


def lif_constant_current_analytic(
    I_const: float,
    t_ms: np.ndarray,
    v0: float = V_REST,
) -> np.ndarray:
    """
    Membrane potential under constant I with no spikes (subthreshold).

    ODE: tau * dV/dt = (V_rest - V) + R*I  => steady state V_ss = V_rest + R*I
    Solution: V(t) = V_ss + (v0 - V_ss) * exp(-t/tau)
    """
    v_ss = V_REST + R_MEM * I_const
    return v_ss + (v0 - v_ss) * np.exp(-t_ms / TAU_MS)


def unpack_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """weights: (..., 220) -> W_ih (..., 10, 20), W_ho (..., 20, 1)."""
    w = np.asarray(weights, dtype=np.float64)
    w1 = w[..., : N_INPUT * N_HIDDEN].reshape(*w.shape[:-1], N_INPUT, N_HIDDEN)
    w2 = w[..., N_INPUT * N_HIDDEN :].reshape(*w.shape[:-1], N_HIDDEN, N_OUTPUT)
    return w1, w2


class LIFState:
    """Vectorized LIF layer state for shape (batch, n_neurons)."""

    __slots__ = ("v", "refrac")

    def __init__(self, batch: int, n: int, rng: np.random.Generator | None = None):
        self.v = np.full((batch, n), V_REST, dtype=np.float64)
        self.refrac = np.zeros((batch, n), dtype=np.int32)
        if rng is not None:
            self.v = rng.normal(V_REST, 0.01, size=(batch, n)).astype(np.float64)

    def step(self, current: np.ndarray) -> np.ndarray:
        """
        current: (batch, n) — synaptic drive I[t] in the membrane equation.

        Returns spikes (batch, n) as float 0/1 for downstream use.
        """
        blocked = self.refrac > 0
        dv = DT_OVER_TAU * (V_REST - self.v) + DT_OVER_TAU * R_MEM * current
        self.v = np.where(~blocked, self.v + dv, self.v)

        spike = (self.v >= V_THRESHOLD) & ~blocked
        self.v = np.where(spike, V_RESET, self.v)
        self.refrac = np.where(blocked, self.refrac - 1, self.refrac)
        self.refrac = np.where(spike, REFRACTORY_STEPS, self.refrac)
        return spike.astype(np.float64)


class NetworkBatch:
    """
    Batched 10→20→1 networks: weights (N, 220), same input spikes for all agents.
    """

    __slots__ = ("n_agents", "w_ih", "w_ho", "hidden", "out")

    def __init__(self, weights: np.ndarray, rng: np.random.Generator | None = None):
        """
        weights: (N, 220)
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != N_WEIGHTS:
            raise ValueError(f"weights must be (N, {N_WEIGHTS}), got {w.shape}")
        self.n_agents = w.shape[0]
        self.w_ih, self.w_ho = unpack_weights(w)
        self.hidden = LIFState(self.n_agents, N_HIDDEN, rng)
        self.out = LIFState(self.n_agents, N_OUTPUT, rng)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self.hidden = LIFState(self.n_agents, N_HIDDEN, rng)
        self.out = LIFState(self.n_agents, N_OUTPUT, rng)

    def step(self, input_spikes: np.ndarray) -> np.ndarray:
        """
        input_spikes: (10,) values in {0,1}

        Returns output spikes (N, 1) as float.
        """
        s = np.asarray(input_spikes, dtype=np.float64).reshape(N_INPUT)
        # I_h = I_in * sum_i W_ij * s_i  -> (N, 20)
        ih = I_IN * (self.w_ih * s.reshape(1, N_INPUT, 1)).sum(axis=1)
        h_spike = self.hidden.step(ih)
        # I_o = I_in * sum_j W_j * h_j
        io = I_IN * (self.w_ho * h_spike.reshape(self.n_agents, N_HIDDEN, 1)).sum(axis=1)
        return self.out.step(io)


class DirectLIF10to1:
    """
    One LIF output driven by 10 inputs with per-synapse weights (Gate A / calibration).
    I = I_in * sum_i w_i * s_i
    """

    __slots__ = ("w", "out")

    def __init__(self, weights_input: np.ndarray):
        w = np.asarray(weights_input, dtype=np.float64).reshape(N_INPUT)
        self.w = w
        self.out = LIFState(1, N_OUTPUT)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self.out = LIFState(1, N_OUTPUT, rng)

    def step(self, input_spikes: np.ndarray) -> float:
        s = np.asarray(input_spikes, dtype=np.float64).reshape(N_INPUT)
        i_sum = float(np.dot(self.w, s))
        I = I_IN * i_sum
        return float(self.out.step(np.array([[I]], dtype=np.float64))[0, 0])


class NetworkSingle:
    """Single network (N=1) convenience wrapper."""

    def __init__(self, weights: np.ndarray, rng: np.random.Generator | None = None):
        w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
        self._batch = NetworkBatch(w, rng)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        self._batch.reset(rng)

    def step(self, input_spikes: np.ndarray) -> float:
        return float(self._batch.step(input_spikes)[0, 0])
