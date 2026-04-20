"""Population dynamics, global σ, births/deaths/replacement (SPEC §1, §5)."""

from __future__ import annotations

import numpy as np

from .agent import N_HISTORY, pearson_r
from .economy import (
    BREATH_PER_WINDOW,
    BUDGET_MODE_NONE,
    BUDGET_MODE_SHARED,
    C_COST_REPRO,
    C_INIT,
    C_REPRO,
    R_MAX,
    VALID_BUDGET_MODES,
    plain_rewards,
    shared_budget_rewards,
)
from .neuron import (
    N_HIDDEN,
    N_INPUT,
    N_OUTPUT,
    N_WEIGHTS,
    V_REST,
    NetworkBatch,
)
from .stimulus import draw_input_rate, poisson_spikes_window

SIGMA_BASE = 0.3


def gini_coefficient(x: np.ndarray) -> float:
    """
    Gini for non-negative values, in [0, 1]; empty -> 0.

    For sorted ascending y_1..y_n with sum S:
        G = (2 * sum(i * y_i)) / (n * S) - (n + 1) / n
    """
    v = np.asarray(x, dtype=np.float64)
    v = v[v > 0] if np.any(v > 0) else v
    if v.size == 0:
        return 0.0
    v = np.sort(np.abs(v))
    n = v.size
    s = v.sum()
    if s <= 0.0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * (idx * v).sum() / s - (n + 1)) / n
    return float(max(0.0, min(1.0, g)))


class Population:
    """Slot-backed population for vectorized simulation."""

    __slots__ = (
        "pop_max",
        "rng",
        "alive",
        "weights",
        "credit",
        "_hc",
        "_fin",
        "_fout",
        "v_h",
        "ref_h",
        "v_o",
        "ref_o",
        "carrying_capacity",
        "budget_mode",
    )

    def __init__(
        self,
        pop_max: int,
        rng: np.random.Generator,
        n_initial: int | None = None,
        carrying_capacity: int | None = None,
        budget_mode: str = BUDGET_MODE_NONE,
    ):
        self.pop_max = int(pop_max)
        self.rng = rng
        n0 = self.pop_max if n_initial is None else min(int(n_initial), self.pop_max)
        if budget_mode not in VALID_BUDGET_MODES:
            raise ValueError(
                f"budget_mode must be one of {VALID_BUDGET_MODES}, got {budget_mode!r}"
            )
        self.budget_mode = str(budget_mode)
        self.carrying_capacity = int(carrying_capacity) if carrying_capacity else 0
        if self.budget_mode == BUDGET_MODE_SHARED and self.carrying_capacity <= 0:
            raise ValueError("budget_mode='shared' requires carrying_capacity > 0")
        self.alive = np.zeros(self.pop_max, dtype=bool)
        self.weights = np.zeros((self.pop_max, N_WEIGHTS), dtype=np.float64)
        self.credit = np.zeros(self.pop_max, dtype=np.float64)
        self._hc = np.zeros(self.pop_max, dtype=np.int64)
        self._fin = np.zeros((self.pop_max, N_HISTORY), dtype=np.float64)
        self._fout = np.zeros((self.pop_max, N_HISTORY), dtype=np.float64)
        self.v_h = np.full((self.pop_max, N_HIDDEN), V_REST, dtype=np.float64)
        self.ref_h = np.zeros((self.pop_max, N_HIDDEN), dtype=np.int32)
        self.v_o = np.full((self.pop_max, N_OUTPUT), V_REST, dtype=np.float64)
        self.ref_o = np.zeros((self.pop_max, N_OUTPUT), dtype=np.int32)

        for i in range(n0):
            self.spawn_initial_slot(i)

    def spawn_initial_slot(self, slot: int) -> None:
        self.alive[slot] = True
        self.weights[slot] = self.rng.uniform(-3.0, 3.0, size=N_WEIGHTS)
        self.credit[slot] = C_INIT
        self._hc[slot] = 0
        self._fin[slot].fill(0.0)
        self._fout[slot].fill(0.0)
        self.v_h[slot].fill(V_REST)
        self.ref_h[slot].fill(0)
        self.v_o[slot].fill(V_REST)
        self.ref_o[slot].fill(0)

    def living_indices(self) -> np.ndarray:
        return np.flatnonzero(self.alive)

    def n_living(self) -> int:
        return int(self.alive.sum())

    def _record_window(self, slot: int, f_in: float, f_out: float) -> None:
        c = int(self._hc[slot])
        idx = c % N_HISTORY
        self._fin[slot, idx] = f_in
        self._fout[slot, idx] = f_out
        self._hc[slot] = c + 1

    def _fitness_slot(self, slot: int) -> float:
        c = int(self._hc[slot])
        if c < N_HISTORY:
            return 0.0
        base = c - N_HISTORY
        idx = (np.arange(N_HISTORY, dtype=np.int64) + base) % N_HISTORY
        fi = self._fin[slot, idx]
        fo = self._fout[slot, idx]
        return pearson_r(fi, fo)

    def _fitness_defined(self, slot: int) -> bool:
        return int(self._hc[slot]) >= N_HISTORY

    def global_sigma(self) -> float:
        idx = self.living_indices()
        rs: list[float] = []
        for s in idx.tolist():
            if self._fitness_defined(int(s)):
                rs.append(self._fitness_slot(int(s)))
        if not rs:
            mean_f = 0.0
        else:
            mean_f = float(np.mean(rs))
        return float(SIGMA_BASE * np.exp(-2.0 * max(0.0, mean_f)))

    def _reset_membrane_slot(self, slot: int) -> None:
        self.v_h[slot].fill(V_REST)
        self.ref_h[slot].fill(0)
        self.v_o[slot].fill(V_REST)
        self.ref_o[slot].fill(0)

    def _run_network_window(self, idx: np.ndarray, spikes: np.ndarray) -> np.ndarray:
        """Returns output spike counts per alive agent (shape n_alive,)."""
        if idx.size == 0:
            return np.zeros(0, dtype=np.int64)
        W = self.weights[idx]
        net = NetworkBatch(W)
        net.hidden.v[...] = self.v_h[idx]
        net.hidden.refrac[...] = self.ref_h[idx]
        net.out.v[...] = self.v_o[idx]
        net.out.refrac[...] = self.ref_o[idx]
        n_steps = spikes.shape[0]
        counts = np.zeros(idx.size, dtype=np.int64)
        for t in range(n_steps):
            o = net.step(spikes[t])
            counts += o[:, 0].astype(np.int64)
        self.v_h[idx] = net.hidden.v
        self.ref_h[idx] = net.hidden.refrac
        self.v_o[idx] = net.out.v
        self.ref_o[idx] = net.out.refrac
        return counts

    def _find_free_slot(self) -> int | None:
        for s in range(self.pop_max):
            if not self.alive[s]:
                return int(s)
        return None

    def _pick_replacement_victim(self, parent: int) -> int:
        """Lowest Credit among alive != parent; tie-break lowest fitness then index (SPEC §5.4)."""
        idx = self.living_indices()
        cand = idx[idx != parent]
        if cand.size == 0:
            cand = idx
        cred = self.credit[cand]
        fit = np.array(
            [self._fitness_slot(int(s)) if self._fitness_defined(int(s)) else -1e300 for s in cand.tolist()],
            dtype=np.float64,
        )
        order = np.lexsort((cand.astype(np.int64), fit, cred))
        return int(cand[order[0]])

    def _write_child_into_slot(self, slot: int, w_child: np.ndarray) -> None:
        self.alive[slot] = True
        self.weights[slot] = w_child
        self.credit[slot] = C_INIT
        self._hc[slot] = 0
        self._fin[slot].fill(0.0)
        self._fout[slot].fill(0.0)
        self._reset_membrane_slot(slot)

    def step_window(self) -> dict:
        """
        Advance one 500 ms window. Returns counts/stats for telemetry.
        """
        f_in = draw_input_rate(self.rng)
        spikes = poisson_spikes_window(self.rng, f_in, 500.0, N_INPUT)
        idx = np.sort(self.living_indices())
        counts = self._run_network_window(idx, spikes)
        f_outs = counts.astype(np.float64) / 0.5

        births = 0
        deaths = 0
        repro_parent_slots: list[int] = []
        repro_child_slots: list[int] = []

        for j, slot in enumerate(idx.tolist()):
            self._record_window(slot, f_in, float(f_outs[j]))

        sigma = self.global_sigma()

        n_alive = idx.size
        defined_mask = np.zeros(n_alive, dtype=bool)
        r_vals_arr = np.zeros(n_alive, dtype=np.float64)
        for j, slot in enumerate(idx.tolist()):
            if self._fitness_defined(slot):
                defined_mask[j] = True
                r_vals_arr[j] = self._fitness_slot(slot)

        if self.budget_mode == BUDGET_MODE_SHARED and self.carrying_capacity > 0:
            rewards, budget_pressure = shared_budget_rewards(
                defined_mask, r_vals_arr, self.carrying_capacity
            )
        else:
            rewards = plain_rewards(defined_mask, r_vals_arr)
            budget_pressure = 0.0  # sentinel: budget mode disabled

        if n_alive > 0:
            self.credit[idx] = self.credit[idx] + rewards - BREATH_PER_WINDOW

        if defined_mask.any():
            r_def = r_vals_arr[defined_mask]
            r_max = float(r_def.max())
            r_mean = float(r_def.mean())
        else:
            r_max = 0.0
            r_mean = 0.0

        dead_slots: list[int] = []
        for slot in idx.tolist():
            if self.credit[slot] <= 0.0:
                dead_slots.append(slot)
        for slot in dead_slots:
            self.alive[slot] = False
            deaths += 1
            self._reset_membrane_slot(slot)

        idx_after_death = np.sort(self.living_indices())
        parents: list[int] = []
        for slot in idx_after_death.tolist():
            if self.credit[slot] >= C_REPRO:
                parents.append(slot)

        parents.sort()

        for parent in parents:
            if not self.alive[parent]:
                continue
            if self.credit[parent] < C_REPRO:
                continue
            self.credit[parent] -= C_COST_REPRO
            w_child = self.weights[parent] + self.rng.normal(0.0, sigma, size=N_WEIGHTS)
            free = self._find_free_slot()
            if free is not None:
                child_slot = int(free)
                self._write_child_into_slot(child_slot, w_child)
            else:
                victim = self._pick_replacement_victim(parent)
                child_slot = int(victim)
                self._write_child_into_slot(child_slot, w_child)
                deaths += 1
            births += 1
            repro_parent_slots.append(int(parent))
            repro_child_slots.append(child_slot)

        return {
            "f_in": f_in,
            "births": births,
            "deaths": deaths,
            "r_max": r_max,
            "r_mean": r_mean,
            "sigma": sigma,
            "budget_pressure": float(budget_pressure),
            "dead_slots": np.asarray(dead_slots, dtype=np.int32),
            "repro_parent_slots": np.asarray(repro_parent_slots, dtype=np.int32),
            "repro_child_slots": np.asarray(repro_child_slots, dtype=np.int32),
        }

    def weight_diversity_metric(self) -> float:
        """Mean over weight positions of std across living agents (SPEC §8.1)."""
        idx = self.living_indices()
        if idx.size <= 1:
            return 0.0
        W = self.weights[idx]
        return float(np.mean(np.std(W, axis=0)))

    def credit_gini(self) -> float:
        idx = self.living_indices()
        if idx.size == 0:
            return 0.0
        return gini_coefficient(self.credit[idx])

    def credit_mean(self) -> float:
        idx = self.living_indices()
        if idx.size == 0:
            return 0.0
        return float(np.mean(self.credit[idx]))

    def max_fitness(self) -> float:
        m = 0.0
        for s in self.living_indices().tolist():
            if self._fitness_defined(int(s)):
                m = max(m, self._fitness_slot(int(s)))
        return float(m)

    def any_success(self) -> bool:
        return self.max_fitness() >= 0.7

    def top_k_slots(self, k: int, by: str = "fitness") -> np.ndarray:
        idx = self.living_indices()
        if idx.size == 0:
            return idx
        if by == "credit":
            scores = self.credit[idx]
        else:
            scores = np.array(
                [
                    self._fitness_slot(int(s)) if self._fitness_defined(int(s)) else -1e300
                    for s in idx.tolist()
                ]
            )
        order = np.argsort(scores, kind="stable")[::-1]
        pick = order[: min(k, order.size)]
        return idx[pick]
