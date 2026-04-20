"""Single-agent wrapper: weights, Credit, rolling fitness history (SPEC §4)."""

from __future__ import annotations

import numpy as np

from .neuron import N_WEIGHTS, NetworkSingle


N_HISTORY = 40


def pearson_r(f_in: np.ndarray, f_out: np.ndarray) -> float:
    """Pearson r; returns 0.0 if undefined or zero output variance (SPEC §4.1)."""
    x = np.asarray(f_in, dtype=np.float64)
    y = np.asarray(f_out, dtype=np.float64)
    if x.size < N_HISTORY or y.size < N_HISTORY:
        return 0.0
    if np.var(y) == 0.0:
        return 0.0
    xm = x.mean()
    ym = y.mean()
    dx = x - xm
    dy = y - ym
    denom = np.sqrt((dx * dx).sum()) * np.sqrt((dy * dy).sum())
    if denom == 0.0:
        return 0.0
    return float((dx * dy).sum() / denom)


class Agent:
    """One evolvable SNN agent (topology fixed in L1)."""

    __slots__ = ("weights", "credit", "_fin", "_fout", "_count", "network")

    def __init__(
        self,
        weights: np.ndarray,
        credit: float = 50.0,
        rng: np.random.Generator | None = None,
    ):
        self.weights = np.asarray(weights, dtype=np.float64).reshape(N_WEIGHTS)
        self.credit = float(credit)
        self._fin = np.zeros(N_HISTORY, dtype=np.float64)
        self._fout = np.zeros(N_HISTORY, dtype=np.float64)
        self._count = 0
        self.network = NetworkSingle(self.weights, rng)

    def reset_episode(self, rng: np.random.Generator | None = None) -> None:
        """Reset membrane (e.g. new run); keeps weights and credit."""
        self.network.reset(rng)

    def clear_history(self) -> None:
        self._count = 0
        self._fin.fill(0.0)
        self._fout.fill(0.0)

    def record_window(self, f_in: float, f_out: float) -> None:
        """Append one (f_in, f_out) sample for this 500 ms window."""
        idx = self._count % N_HISTORY
        self._fin[idx] = f_in
        self._fout[idx] = f_out
        self._count += 1

    def fitness_defined(self) -> bool:
        return self._count >= N_HISTORY

    def _last40(self) -> tuple[np.ndarray, np.ndarray]:
        base = self._count - N_HISTORY
        idx = (np.arange(N_HISTORY, dtype=np.int64) + base) % N_HISTORY
        return self._fin[idx].copy(), self._fout[idx].copy()

    def fitness(self) -> float:
        if not self.fitness_defined():
            return 0.0
        fi, fo = self._last40()
        return pearson_r(fi, fo)

    def reward_delta(self) -> float:
        """ΔCredit_reward = max(0, r) * R_max (SPEC §4.2)."""
        r = self.fitness() if self.fitness_defined() else 0.0
        return max(0.0, r) * 5.0
