"""Population dynamics, global σ, births/deaths/replacement (SPEC §1, §5)."""

from __future__ import annotations

import numpy as np

from .agent import N_HISTORY, fitness_with_calibration_penalty, pearson_r
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
from .slime import (
    SlimeConfig,
    blend_weights,
    chemotaxis_step,
    decay_and_diffuse,
    emit,
    hgt_pairs,
    new_field,
    position_near,
    random_positions,
    reward_bonus,
    sense,
)
from .stimulus import draw_input_rate, poisson_spikes_window
from .oracle import (
    classify_output,
    draw_oracle_sample,
    poisson_three_channels,
    MODE_AND,
    MODE_NOT,
)
from .task import DEFAULT_TASK, TASK_L1, TASK_L2V2, is_logic_task, validate_task

SIGMA_BASE = 0.3

# L2v2 weight initialisation: SPEC §3.1 — inhibitory weights enabled.
# Symmetric uniform around 0 already gives ~50% negative weights, well above
# the SPEC's "≥20% negative" lower bound, so we don't need extra biasing.
L2V2_WEIGHT_INIT_LOW = -1.5
L2V2_WEIGHT_INIT_HIGH = 1.5

# Per-mode rolling correctness window (in 500 ms windows).  Smaller than
# N_HISTORY because logic windows are noisy; 20 keeps the bar reactive.
LOGIC_HISTORY = 20


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
        "slime",
        "positions",
        "pheromone",
        "_last_local_p",
        "_last_field_max",
        "_last_hgt_count",
        "_last_migrations",
        "_last_hgt_pairs",
        "_last_reward",
        "_last_credit_delta",
        "calibration_lambda",
        "synapse_gain",
        "task",
        # L2v2 logic-task per-agent rolling correctness ring buffers
        "_acc_and_hits",
        "_acc_and_n",
        "_acc_not_hits",
        "_acc_not_n",
        "_acc_and_buf",
        "_acc_not_buf",
        # L2v2 last-window snapshot for telemetry
        "_last_oracle",
        "_last_consensus_bit",
        "_last_consensus_acc",
        "_last_acc_and_pop",
        "_last_acc_not_pop",
        "_last_both_pass_pct",
        "_last_logic_diversity",
    )

    def __init__(
        self,
        pop_max: int,
        rng: np.random.Generator,
        n_initial: int | None = None,
        carrying_capacity: int | None = None,
        budget_mode: str = BUDGET_MODE_NONE,
        slime: SlimeConfig | None = None,
        calibration_lambda: float = 0.0,
        synapse_gain: float = 1.0,
        task: str = DEFAULT_TASK,
    ):
        self.pop_max = int(pop_max)
        self.rng = rng
        self.task = validate_task(task)
        n0 = self.pop_max if n_initial is None else min(int(n_initial), self.pop_max)
        if budget_mode not in VALID_BUDGET_MODES:
            raise ValueError(
                f"budget_mode must be one of {VALID_BUDGET_MODES}, got {budget_mode!r}"
            )
        self.budget_mode = str(budget_mode)
        self.carrying_capacity = int(carrying_capacity) if carrying_capacity else 0
        if self.budget_mode == BUDGET_MODE_SHARED and self.carrying_capacity <= 0:
            raise ValueError("budget_mode='shared' requires carrying_capacity > 0")
        self.slime = slime if slime is not None else SlimeConfig()
        self.slime.validate()
        if calibration_lambda < 0.0:
            raise ValueError("calibration_lambda must be ≥ 0.0")
        self.calibration_lambda = float(calibration_lambda)
        if synapse_gain <= 0.0:
            raise ValueError("synapse_gain must be > 0.0")
        self.synapse_gain = float(synapse_gain)
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

        # Slime spatial state (only used when self.slime.enabled).
        self.positions = np.zeros((self.pop_max, 2), dtype=np.int32)
        self.pheromone = new_field(self.slime.grid_size)
        self._last_local_p = np.zeros(0, dtype=np.float64)
        self._last_field_max = 0.0
        self._last_hgt_count = 0
        self._last_migrations = 0
        self._last_hgt_pairs: list[tuple[int, int]] = []
        self._last_reward = np.zeros(self.pop_max, dtype=np.float64)
        self._last_credit_delta = np.zeros(self.pop_max, dtype=np.float64)

        # L2v2 logic accuracy ring buffers (per agent, separate AND / NOT).
        self._acc_and_buf = np.zeros((self.pop_max, LOGIC_HISTORY), dtype=np.int8)
        self._acc_not_buf = np.zeros((self.pop_max, LOGIC_HISTORY), dtype=np.int8)
        self._acc_and_n = np.zeros(self.pop_max, dtype=np.int32)
        self._acc_not_n = np.zeros(self.pop_max, dtype=np.int32)
        self._acc_and_hits = np.zeros(self.pop_max, dtype=np.int32)
        self._acc_not_hits = np.zeros(self.pop_max, dtype=np.int32)
        self._last_oracle = None  # OracleSample | None
        self._last_consensus_bit: int | None = None
        self._last_consensus_acc: float = 0.0
        self._last_acc_and_pop: float = 0.0
        self._last_acc_not_pop: float = 0.0
        self._last_both_pass_pct: float = 0.0
        self._last_logic_diversity: float = 0.0

        if self.slime.enabled:
            init_xy = random_positions(self.rng, self.pop_max, self.slime.grid_size)
            self.positions[...] = init_xy

        for i in range(n0):
            self.spawn_initial_slot(i)

    def _init_weights_for_task(self) -> np.ndarray:
        """Sample one fresh weight vector with the task-appropriate range.

        L1   : Uniform(-3, 3)        — SPEC §1.1.
        L2v2 : Uniform(-1.5, 1.5)    — SPEC_L2_V2.0 §3.1 inhibitory weights
                                       (symmetric => ~50% negative, satisfies
                                       the SPEC's "≥20% negative" requirement).
        """
        if self.task == TASK_L2V2:
            return self.rng.uniform(
                L2V2_WEIGHT_INIT_LOW, L2V2_WEIGHT_INIT_HIGH, size=N_WEIGHTS
            )
        return self.rng.uniform(-3.0, 3.0, size=N_WEIGHTS)

    def _reset_logic_buffers_slot(self, slot: int) -> None:
        self._acc_and_buf[slot].fill(0)
        self._acc_not_buf[slot].fill(0)
        self._acc_and_n[slot] = 0
        self._acc_not_n[slot] = 0
        self._acc_and_hits[slot] = 0
        self._acc_not_hits[slot] = 0

    def spawn_initial_slot(self, slot: int) -> None:
        self.alive[slot] = True
        self.weights[slot] = self._init_weights_for_task()
        self.credit[slot] = C_INIT
        self._hc[slot] = 0
        self._fin[slot].fill(0.0)
        self._fout[slot].fill(0.0)
        self.v_h[slot].fill(V_REST)
        self.ref_h[slot].fill(0)
        self.v_o[slot].fill(V_REST)
        self.ref_o[slot].fill(0)
        self._reset_logic_buffers_slot(slot)

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
        if self.task == TASK_L2V2:
            return self._logic_fitness_slot(slot)
        c = int(self._hc[slot])
        if c < N_HISTORY:
            return 0.0
        base = c - N_HISTORY
        idx = (np.arange(N_HISTORY, dtype=np.int64) + base) % N_HISTORY
        fi = self._fin[slot, idx]
        fo = self._fout[slot, idx]
        if self.calibration_lambda > 0.0:
            return fitness_with_calibration_penalty(
                fi, fo, self.calibration_lambda
            )
        return pearson_r(fi, fo)

    def _fitness_defined(self, slot: int) -> bool:
        if self.task == TASK_L2V2:
            # In L2v2 we treat fitness as "defined" once *both* mode buffers
            # have at least one sample — otherwise a brand-new agent that has
            # only seen AND windows would be ranked as if it could also do NOT.
            return bool(
                int(self._acc_and_n[slot]) > 0 and int(self._acc_not_n[slot]) > 0
            )
        return int(self._hc[slot]) >= N_HISTORY

    def _logic_acc_slot(self, slot: int, mode: int) -> float:
        if mode == MODE_AND:
            n = int(self._acc_and_n[slot])
            return (int(self._acc_and_hits[slot]) / n) if n > 0 else 0.0
        n = int(self._acc_not_n[slot])
        return (int(self._acc_not_hits[slot]) / n) if n > 0 else 0.0

    def _logic_fitness_slot(self, slot: int) -> float:
        """Mean accuracy across AND and NOT, in [0, 1]."""
        return 0.5 * (self._logic_acc_slot(slot, MODE_AND) + self._logic_acc_slot(slot, MODE_NOT))

    def _record_logic_window(self, slot: int, mode: int, correct: bool) -> None:
        if mode == MODE_AND:
            n = int(self._acc_and_n[slot])
            idx = n % LOGIC_HISTORY
            old = int(self._acc_and_buf[slot, idx])
            self._acc_and_buf[slot, idx] = 1 if correct else 0
            if n >= LOGIC_HISTORY:
                self._acc_and_hits[slot] = int(self._acc_and_hits[slot]) - old + (1 if correct else 0)
            else:
                self._acc_and_hits[slot] = int(self._acc_and_hits[slot]) + (1 if correct else 0)
                self._acc_and_n[slot] = n + 1
        else:
            n = int(self._acc_not_n[slot])
            idx = n % LOGIC_HISTORY
            old = int(self._acc_not_buf[slot, idx])
            self._acc_not_buf[slot, idx] = 1 if correct else 0
            if n >= LOGIC_HISTORY:
                self._acc_not_hits[slot] = int(self._acc_not_hits[slot]) - old + (1 if correct else 0)
            else:
                self._acc_not_hits[slot] = int(self._acc_not_hits[slot]) + (1 if correct else 0)
                self._acc_not_n[slot] = n + 1

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
        net = NetworkBatch(W, output_gain=self.synapse_gain)
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

    def _write_child_into_slot(
        self, slot: int, w_child: np.ndarray, parent_slot: int | None = None
    ) -> None:
        self.alive[slot] = True
        self.weights[slot] = w_child
        self.credit[slot] = C_INIT
        self._hc[slot] = 0
        self._fin[slot].fill(0.0)
        self._fout[slot].fill(0.0)
        self._reset_membrane_slot(slot)
        self._reset_logic_buffers_slot(slot)
        if self.slime.enabled:
            if parent_slot is not None:
                self.positions[slot] = position_near(
                    self.rng, self.positions[parent_slot], self.slime.grid_size, max_offset=1
                )
            else:
                self.positions[slot] = random_positions(self.rng, 1, self.slime.grid_size)[0]

    def step_window(self) -> dict:
        """
        Advance one 500 ms window. Returns counts/stats for telemetry.

        Behaviour switches on ``self.task``:
          - L1       : SPEC §3.1 single-channel rate, Pearson-r reward
          - L2v2_ctrl: SPEC_L2_V2.0 §2 three-channel oracle, table reward
        """
        oracle = None
        if self.task == TASK_L2V2:
            oracle = draw_oracle_sample(self.rng)
            spikes = poisson_three_channels(
                self.rng, oracle.f_a_hz, oracle.f_b_hz, oracle.f_s_hz, 500.0
            )
            f_in = float(oracle.f_a_hz)  # surface a representative scalar for legacy logging
        else:
            f_in = draw_input_rate(self.rng)
            spikes = poisson_spikes_window(self.rng, f_in, 500.0, N_INPUT)

        idx = np.sort(self.living_indices())
        counts = self._run_network_window(idx, spikes)
        f_outs = counts.astype(np.float64) / 0.5

        births = 0
        deaths = 0
        repro_parent_slots: list[int] = []
        repro_child_slots: list[int] = []

        # Reset per-window per-agent reward telemetry
        self._last_reward.fill(0.0)
        self._last_credit_delta.fill(0.0)

        if self.task == TASK_L2V2:
            # SPEC_L2_V2.0 §2.2 — judge each agent against the oracle truth table.
            assert oracle is not None
            n_alive = idx.size
            n_correct = 0
            n_spiking_correct = 0
            out_bits = np.zeros(n_alive, dtype=np.int8)
            rewards = np.zeros(n_alive, dtype=np.float64)
            for j, slot in enumerate(idx.tolist()):
                ob = classify_output(float(f_outs[j]))
                out_bits[j] = ob
                correct = (ob == oracle.target_bit)
                self._record_logic_window(slot, oracle.mode, correct)
                if correct:
                    rewards[j] = oracle.reward_correct
                    n_correct += 1
                else:
                    rewards[j] = oracle.reward_wrong
                # also keep f_in/f_out history defined (zeros) for downstream code paths
                self._record_window(slot, f_in, float(f_outs[j]))
            # Population-wide telemetry: consensus + accuracy
            if n_alive > 0:
                spiking = int(out_bits.sum())
                consensus_bit = 1 if spiking >= (n_alive - spiking) else 0
                consensus_acc = float(n_correct / n_alive)
            else:
                consensus_bit = None
                consensus_acc = 0.0
            self._last_oracle = oracle
            self._last_consensus_bit = consensus_bit
            self._last_consensus_acc = consensus_acc

            # Budget mode is intentionally a no-op for L2v2 — the reward table
            # is already the entire economy.  We still set a sentinel.
            budget_pressure = 0.0
            defined_mask = np.zeros(n_alive, dtype=bool)
            r_vals_arr = np.zeros(n_alive, dtype=np.float64)
            for j, slot in enumerate(idx.tolist()):
                if self._fitness_defined(slot):
                    defined_mask[j] = True
                    r_vals_arr[j] = self._fitness_slot(slot)

            sigma = self.global_sigma()
        else:
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
                budget_pressure = 0.0

        # ── Slime: pheromone-modulated reward (cooperation incentive) ──
        local_p = np.zeros(n_alive, dtype=np.float64)
        field_max = 0.0
        if self.slime.enabled and n_alive > 0:
            local_p = sense(self.pheromone, self.positions[idx])
            field_max = float(self.pheromone.max())
            bonus = reward_bonus(local_p, field_max, self.slime.pheromone_bonus_k)
            rewards = rewards * bonus
        self._last_local_p = local_p
        self._last_field_max = field_max

        if n_alive > 0:
            delta = rewards - BREATH_PER_WINDOW
            self.credit[idx] = self.credit[idx] + delta
            self._last_reward[idx] = rewards
            self._last_credit_delta[idx] = delta

        if defined_mask.any():
            r_def = r_vals_arr[defined_mask]
            r_max = float(r_def.max())
            r_mean = float(r_def.mean())
        else:
            r_max = 0.0
            r_mean = 0.0

        # ── Slime: horizontal gene transfer (social, lateral learning) ──
        hgt_count = 0
        hgt_pair_log: list[tuple[int, int]] = []
        if (
            self.slime.enabled
            and self.slime.hgt_enabled
            and n_alive >= 2
            and self.slime.hgt_prob > 0.0
        ):
            pos_alive = self.positions[idx]
            cred_alive = self.credit[idx].copy()
            pairs = hgt_pairs(
                self.rng,
                pos_alive,
                cred_alive,
                self.slime.grid_size,
                self.slime.hgt_radius,
                self.slime.hgt_prob,
                self.slime.hgt_donor_ratio,
            )
            for r_local, d_local in pairs:
                r_slot = int(idx[r_local])
                d_slot = int(idx[d_local])
                if not self.alive[r_slot] or not self.alive[d_slot]:
                    continue
                if self.credit[r_slot] <= self.slime.hgt_cost:
                    continue
                self.weights[r_slot] = blend_weights(
                    self.weights[r_slot], self.weights[d_slot], self.slime.hgt_blend
                )
                self.credit[r_slot] -= self.slime.hgt_cost
                # Recipient was modified — flush its history; old (f_in, f_out) no longer
                # reflects current weights. Resets fitness to "undefined" until 40 new windows.
                self._hc[r_slot] = 0
                self._fin[r_slot].fill(0.0)
                self._fout[r_slot].fill(0.0)
                # Logic accuracy buffers also no longer reflect the post-HGT weights.
                self._reset_logic_buffers_slot(r_slot)
                hgt_count += 1
                hgt_pair_log.append((r_slot, d_slot))
        self._last_hgt_count = hgt_count
        self._last_hgt_pairs = hgt_pair_log

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
                self._write_child_into_slot(child_slot, w_child, parent_slot=parent)
            else:
                victim = self._pick_replacement_victim(parent)
                child_slot = int(victim)
                self._write_child_into_slot(child_slot, w_child, parent_slot=parent)
                deaths += 1
            births += 1
            repro_parent_slots.append(int(parent))
            repro_child_slots.append(child_slot)

        # ── Slime: pheromone field tick + chemotaxis ──
        migrations = 0
        if self.slime.enabled:
            idx_now = np.sort(self.living_indices())
            if idx_now.size > 0:
                fit_now = np.zeros(idx_now.size, dtype=np.float64)
                for j, slot in enumerate(idx_now.tolist()):
                    if self._fitness_defined(slot):
                        fit_now[j] = self._fitness_slot(slot)
                emit(
                    self.pheromone,
                    self.positions[idx_now],
                    fit_now,
                    self.slime.pheromone_emit,
                )
            self.pheromone = decay_and_diffuse(
                self.pheromone, self.slime.pheromone_decay, self.slime.pheromone_diffusion
            )
            if self.slime.migrate_enabled and idx_now.size > 0:
                old_pos = self.positions[idx_now].copy()
                new_pos = chemotaxis_step(
                    self.rng,
                    self.pheromone,
                    self.positions[idx_now],
                    self.slime.grid_size,
                    self.slime.migrate_prob,
                )
                self.positions[idx_now] = new_pos
                migrations = int(np.any(new_pos != old_pos, axis=1).sum())
        self._last_migrations = migrations

        # ── L2v2 population-level logic metrics (cheap; computed every window) ──
        if self.task == TASK_L2V2:
            alive_idx = self.living_indices()
            if alive_idx.size > 0:
                acc_and_pop = float(
                    np.mean([self._logic_acc_slot(int(s), MODE_AND) for s in alive_idx.tolist()])
                )
                acc_not_pop = float(
                    np.mean([self._logic_acc_slot(int(s), MODE_NOT) for s in alive_idx.tolist()])
                )
                # SPEC §5.3: "5% of elite agents pass both AND and NOT"
                # → fraction of agents that have both acc_AND and acc_NOT ≥ 0.7
                pass_threshold = 0.7
                both_pass = sum(
                    1
                    for s in alive_idx.tolist()
                    if (
                        self._logic_acc_slot(int(s), MODE_AND) >= pass_threshold
                        and self._logic_acc_slot(int(s), MODE_NOT) >= pass_threshold
                    )
                )
                both_pass_pct = float(both_pass / alive_idx.size)
                # Logic_Diversity_Score (SPEC §4.1 deployment note):
                #   1 - |acc_AND - acc_NOT| / max(acc_AND, acc_NOT, ε)
                # → 1 when the population is balanced across the two tasks,
                #   → 0 when it is fully specialised on one (caught in the trap).
                eps = 1e-6
                denom = max(acc_and_pop, acc_not_pop, eps)
                logic_diversity = float(
                    1.0 - abs(acc_and_pop - acc_not_pop) / denom
                )
            else:
                acc_and_pop = 0.0
                acc_not_pop = 0.0
                both_pass_pct = 0.0
                logic_diversity = 0.0
            self._last_acc_and_pop = acc_and_pop
            self._last_acc_not_pop = acc_not_pop
            self._last_both_pass_pct = both_pass_pct
            self._last_logic_diversity = logic_diversity

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
            "pheromone_max": float(self._last_field_max),
            "pheromone_mean": float(self.pheromone.mean()) if self.slime.enabled else 0.0,
            "hgt_count": int(self._last_hgt_count),
            "hgt_pairs": list(self._last_hgt_pairs),
            "migrations": int(migrations),
            # L2v2 (None / zero when task=l1)
            "task": self.task,
            "oracle": (
                {
                    "mode": int(self._last_oracle.mode),
                    "mode_name": self._last_oracle.mode_name,
                    "bit_a": int(self._last_oracle.bit_a),
                    "bit_b": int(self._last_oracle.bit_b),
                    "f_a_hz": float(self._last_oracle.f_a_hz),
                    "f_b_hz": float(self._last_oracle.f_b_hz),
                    "f_s_hz": float(self._last_oracle.f_s_hz),
                    "target_bit": int(self._last_oracle.target_bit),
                    "reward_correct": float(self._last_oracle.reward_correct),
                }
                if (self.task == TASK_L2V2 and self._last_oracle is not None)
                else None
            ),
            "consensus_bit": (
                self._last_consensus_bit if self.task == TASK_L2V2 else None
            ),
            "consensus_acc": (
                float(self._last_consensus_acc) if self.task == TASK_L2V2 else 0.0
            ),
            "acc_and_pop": float(self._last_acc_and_pop) if self.task == TASK_L2V2 else 0.0,
            "acc_not_pop": float(self._last_acc_not_pop) if self.task == TASK_L2V2 else 0.0,
            "both_pass_pct": float(self._last_both_pass_pct) if self.task == TASK_L2V2 else 0.0,
            "logic_diversity": float(self._last_logic_diversity) if self.task == TASK_L2V2 else 0.0,
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
        # In L2v2 fitness is mean(acc_AND, acc_NOT) ∈ [0, 1]; 0.7 ≈ "elite both-pass".
        # In L1 fitness is Pearson r ∈ [-1, 1]; 0.7 is the SPEC §6 success threshold.
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
