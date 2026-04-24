"""Population dynamics, global σ, births/deaths/replacement (SPEC §1, §5)."""

from __future__ import annotations

from dataclasses import dataclass

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
    DEFAULT_TASK_DIFFICULTY,
    MODE_AND,
    MODE_NOT,
    classify_output,
    difficulty_weights,
    draw_oracle_sample,
    poisson_three_channels,
    spike_effort_bonus,
)
from .task import (
    DEFAULT_TASK,
    N_CH_A,
    N_CH_B,
    N_CH_S,
    TASK_L1,
    TASK_L2V2,
    is_logic_task,
    validate_task,
)

SIGMA_BASE = 0.3


@dataclass
class FounderInjection:
    """SPEC_L2_V3.0 §1.3 — one entry in an admixture experiment's founder list.

    ``weights`` is a (K, 220) pool (typically the full living snapshot of a
    saved Strain).  ``fraction`` is the share of ``n_initial`` slots that
    should be filled by sampling (with replacement) from this pool.
    """

    weights: np.ndarray  # (K, N_WEIGHTS) — source pool
    fraction: float       # share of n_initial slots, in [0, 1]
    label: str = ""       # purely informational (e.g. "AND-pure-30min")

# L2v2 weight initialisation — ERRATA v2.5 ("prebiotic stage" founder bias)
# ────────────────────────────────────────────────────────────────────────────
# SPEC §3.1 only requires "≥20% negative weights" for inhibitory paths.
# v2.0–v2.4 used symmetric Uniform(-1.5, 1.5) → E[Σw] = 0 → I_o ≈ 0 →
# f_out ≈ 0 for *every* founder.  This makes the colony's evolvability
# strictly zero on a fresh seed: silent agents can't reproduce (net credit
# < 0), no reproduction means no mutation, no mutation means no escape from
# silence — the entire evolutionary loop never starts.
#
# v2.5 introduces a positive offset on the initial range so a fraction of
# founders (~30-40%) emit f_out > OUT_SPIKING_THRESHOLD_HZ on day-zero
# stimuli.  Negative weights still cover ~25% of the range — well above
# the SPEC §3.1 "≥20% negative" lower bound, so inhibitory paths remain
# discoverable by mutation.
#
# Biological framing: this is *not* intelligent design — it's the standard
# evo-devo move of starting an experiment from a *prebiotically-selected*
# founder population (cf. Lenski's LTEE, which begins with a fully-formed
# E. coli, not from random monomers).  We are asking the question
# "can a population that already produces output evolve logic gates?",
# not "can random matter abiogenesise neural computation?" — the latter
# is the abiogenesis problem and not solvable in the time-scales this
# colony runs at.  See docs/project-summary.md §L2.5 for the full rationale.
L2V2_WEIGHT_INIT_LOW = -0.5
L2V2_WEIGHT_INIT_HIGH = 1.5

# ERRATA v3.3 (Path D1) — "anti-follower" pre-evolved seed for not_only.
# ────────────────────────────────────────────────────────────────────────────
# v3.1 tried "inhibitory bias founder" (Uniform(-1.5, +0.5)) — the reasoning
# was: NOT needs negative output coupling, so let evolution start with a
# negative weight budget.  In v3.2 we paired that with a wrong-answer penalty
# so silent agents starve.  But v3.1+v3.2 combined are mathematically
# self-defeating: an inhibition-biased founder is *silent* on every input
# (E[Σw·s] < 0 → I_o < threshold → never fires), and v3.2 then starves
# every silent agent before it can reproduce.  Whole population dies in
# the first few windows.  Diagnosis: v3.1 was fixing the wrong end of the
# pipe.  NOT is not "inhibition everywhere"; NOT is *context-gated routing*
# — A high → suppress output, A low → let a tonic source drive output.
# Random or uniformly-inhibitory founders cannot stumble into this routing
# topology with positive probability in the time-scales the colony runs at.
#
# v3.3 replaces the noise prior with a *structural* prior: every not_only
# founder is born with a hand-designed anti-follower micro-circuit, and
# evolution's job is reduced to refining the magnitudes.  Topology:
#
#   * Hidden 0..H_A_DET-1   = "A-detectors":
#         A→h:  strongly positive   → hidden tracks A channel
#         h→o:  strongly NEGATIVE   → A high suppresses output
#   * Hidden H_A_DET..N_HIDDEN-1 = "S-tonic drivers":
#         S→h:  strongly positive   → S=80Hz drives hidden constantly
#         h→o:  positive            → keeps output spiking unless suppressed
#   * All other input→hidden cells (B paths, cross-specialisation paths)
#     get small symmetric noise → the substrate for mutation to discover
#     refinements (e.g. NOT-on-B, AND-comorbidity).
#
# Net behaviour with this seed at f_s=80Hz:
#   a=1 (75Hz)  →  A-detectors fire   → suppress output  → silent ✓
#   a=0 (25Hz)  →  A-detectors silent → S-tonic drives   → spike  ✓
#
# This is *not* intelligent design in the pejorative sense — it's the same
# prebiotic-selection move as v2.5 (Uniform(-0.5,+1.5) was *also* a
# structured prior, just one that happened to favour excitation), only now
# the structure is targeted at the niche we want to colonise.  Biological
# precedent: GABAergic neurons evolved from glutamatergic ancestors via
# postsynaptic ion-channel polarity flip; ON/OFF retinal ganglion cells
# share architecture but differ at one synapse's polarity.  See
# docs/project-summary.md §5.11 for the full rationale.
#
# Magnitude rationale (LIF V_THRESH=1.0, R=1.0, I_IN=2.5, tau=20ms,
# channels at 25/75/80 Hz, refractory 2ms):
#
# A hidden cell driven by k Poisson channels (each at rate f, weight w)
# sees a per-ms expected current I_avg = I_IN · w · k · f/1000.
# Firing happens when I_avg ≳ V_threshold (=1.0).  For:
#
#   * A→A-detectors (k=4, f=75Hz at A-high):
#         w=2.25 → I_avg = 2.5·2.25·4·0.075 = 1.69 → A-det fires ~50Hz ✓
#         w=2.25 at a-low (f=25Hz):
#             I_avg = 2.5·2.25·4·0.025 = 0.56 → A-det stays quiet ✓
#   * S→S-tonic (k=2, f=80Hz):
#         w=4.0 → I_avg = 2.5·4·2·0.08 = 1.6 → S-tonic fires ~40Hz ✓
#
# A-det→output: 10 A-dets at 50Hz → 0.5 spikes/ms expected.  With
# w=-3.0: contribution = 2.5·(-3.0)·0.5 = -3.75 (heavy suppression).
# S-tonic→output: 10 tonics at 40Hz → 0.4 spikes/ms.  With w=1.75:
# contribution = +1.75 → drives output when A is silent.  Net at
# a=high (-3.75+1.75 = -2.0 → silent), a=low (0+1.75 = +1.75 → fires).
#
# These weights are larger than the v2.5 mixed-dish range (Uniform(-0.5,
# +1.5)) because the structural roles need stronger drive — but that's
# fine: this seed is the FOUNDER of the not_only niche, mutation-
# selection then refines from here.  global_sigma annealing handles
# magnitude-scale exploration.
L2V2_NOT_SEED_STRONG_POS_LOW = 1.5
L2V2_NOT_SEED_STRONG_POS_HIGH = 3.0
L2V2_NOT_SEED_STRONG_NEG_LOW = -4.0
L2V2_NOT_SEED_STRONG_NEG_HIGH = -2.0
L2V2_NOT_SEED_TONIC_POS_LOW = 3.0
L2V2_NOT_SEED_TONIC_POS_HIGH = 5.0
L2V2_NOT_SEED_OUT_TONIC_LOW = 1.0
L2V2_NOT_SEED_OUT_TONIC_HIGH = 2.5
L2V2_NOT_SEED_NOISE_LOW = -0.2
L2V2_NOT_SEED_NOISE_HIGH = 0.2
# Hidden-layer split: half do A-detection, half do S-tonic driving.  Even
# split keeps both pathways equally represented at birth so mutation +
# selection can re-balance per niche.
L2V2_NOT_SEED_H_A_DET = N_HIDDEN // 2

# Difficulty preset name that triggers the anti-follower structural seed.
# Kept as a constant so tests/UI can reference it without string-matching.
DIFFICULTY_ANTI_FOLLOWER_SEED = "not_only"

# SPEC_L2_V3.5 — niche / species classification thresholds.
# An agent counts as a "specialist" in a logic mode once its rolling per-mode
# accuracy is at or above this threshold AND it has accumulated at least
# NICHE_MIN_SAMPLES windows of evidence in that mode.  Below either bar, it's
# classified as NOVICE (untrained / dead-on-arrival) regardless of luck.
NICHE_SPECIALIST_THRESHOLD = 0.65
NICHE_MIN_SAMPLES = 10

# SPEC_L2_V3.5c — "hard-row" qualification for specialists.
# Diagnostic motivation: under task_difficulty=balanced the (1,1)-AND row is
# sampled ~50 % of AND windows, the other three (target=0) are ~50 % combined.
# A 100 %-silent agent therefore scores 50 % on AND mode → just under the 0.65
# bar, but a *half-firing* agent (only fires on (1,1) half the time) scores
# 0.5×0.41 + 0.5×1.0 ≈ 0.70 — *passes* the bar while only half-learning the
# hard row.  v3.5b's swarm metrics correctly reported this (acc_and_11_swarm
# = 41 %), but the species classification still let those silent-pretenders
# count as AND_EXPERT, which dragged the swarm denominator down.
#
# Fix: an agent is only an AND_EXPERT if it can ALSO clear the bar on the
# specifically hard "produce 1" row, i.e. acc on (1,1)-AND windows ≥ this
# threshold over at least NICHE_HARD_MIN_SAMPLES of those windows.  Same for
# NOT (a=0)-rows.  The general per-mode bar still applies on top, so an agent
# that only fires on (1,1) but misses the easy zeros doesn't accidentally
# qualify either.
NICHE_HARD_THRESHOLD = 0.50      # ≥ chance level on the hard "produce 1" row
NICHE_HARD_MIN_SAMPLES = 5       # at least 5 of THAT specific row in history
LOGIC_HARD_HISTORY = 16          # rolling buffer length for the hard rows

# Species labels for the four-quadrant logic-space partition.  Stored as
# small ints so telemetry can ship them as a flat list of 220 bytes.
SPECIES_NOVICE = 0       # acc_AND < threshold AND acc_NOT < threshold
SPECIES_AND_EXPERT = 1   # acc_AND ≥ threshold, acc_NOT < threshold
SPECIES_NOT_EXPERT = 2   # acc_AND < threshold, acc_NOT ≥ threshold
SPECIES_DUAL_EXPERT = 3  # acc_AND ≥ threshold AND acc_NOT ≥ threshold

# Per-mode rolling correctness window (in 500 ms windows).  Smaller than
# N_HISTORY because logic windows are noisy; 20 keeps the bar reactive.
LOGIC_HISTORY = 20

# Population-level row-specific accuracy ring buffer length, in windows.
# Larger than LOGIC_HISTORY because each individual row only fires on a
# fraction of windows (e.g. (1,1) fires P=p_and_target_one × p_mode_and ≈
# 25% of windows in balanced mode).  200 windows ≈ 100 s wall-clock at 20 Hz
# and gives ~50 (1,1) samples to average over.
ROW_HISTORY = 200

# Six row-specific buckets indexed in this order:
#   0 = AND (0,0)   1 = AND (0,1)   2 = AND (1,0)   3 = AND (1,1)
#   4 = NOT a=0     5 = NOT a=1
N_ROW_BUCKETS = 6


def _row_bucket(mode: int, bit_a: int, bit_b: int) -> int:
    """Map (mode, a, b) → 0..5 row bucket index."""
    if mode == MODE_AND:
        return 2 * int(bit_a) + int(bit_b)         # 0..3 for (0,0)/(0,1)/(1,0)/(1,1)
    return 4 + int(bit_a)                          # 4 for NOT a=0; 5 for NOT a=1


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
        # SPEC_L2_V3.5c — per-agent rolling buffer for the "hard" rows only.
        # _acc_and_hard_buf records correct/wrong on (1,1)-AND windows;
        # _acc_not_hard_buf records correct/wrong on (a=0)-NOT windows.
        # Used by _species_of_slot to filter out silent-pretender specialists.
        "_acc_and_hard_buf",
        "_acc_and_hard_n",
        "_acc_and_hard_idx",
        "_acc_not_hard_buf",
        "_acc_not_hard_n",
        "_acc_not_hard_idx",
        # L2v2 last-window snapshot for telemetry
        "_last_oracle",
        "_last_consensus_bit",
        "_last_consensus_acc",
        "_last_acc_and_pop",
        "_last_acc_not_pop",
        "_last_both_pass_pct",
        "_last_logic_diversity",
        # L2v2 task-difficulty (environment shaping) + per-row population telemetry
        "task_difficulty",
        "_difficulty_weights",
        "_row_buf_correct",
        "_row_buf_total",
        "_row_buf_idx",
        "_row_buf_filled",
        "_last_row_acc",
        # SPEC_L2_V3.4 — 3-phase admixture protocol (commensal → exchange → restored)
        "admixture_commensal_s",
        "admixture_exchange_s",
        "admixture_phase2_blend",
        "admixture_phase2_prob_mul",
        "_t_sim_seconds",
        "_needs_and_samples",
        "_needs_not_samples",
        # SPEC_L2_V3.5 — assortative HGT (prezygotic isolation) + swarm-level
        # dual-logic metrics (post-speciation L2 success criterion)
        "assortative_temperature",
        "_last_species_counts",          # {SPECIES_*: int}
        "_last_acc_and_swarm",           # mean acc_AND over AND-or-dual experts
        "_last_acc_not_swarm",           # mean acc_NOT over NOT-or-dual experts
        "_last_colony_dual_acc",         # (acc_and_swarm + acc_not_swarm) / 2 if both species exist
        # SPEC_L2_V3.5b — niche-aware consensus + per-row swarm accuracy.
        # The legacy consensus_acc / acc_*_11_pop average over ALL alive agents,
        # which under speciation gets diluted by the "wrong species correctly
        # silent" cohort.  These swarm versions only count voters whose niche
        # matches the current oracle mode.
        "_last_consensus_acc_swarm",     # accuracy among on-niche voters this window
        "_last_consensus_bit_swarm",     # majority vote among on-niche voters
        "_last_consensus_voters_swarm",  # how many on-niche voters this window
        "_row_buf_correct_swarm",        # parallel row buffers, only on-niche voters
        "_row_buf_total_swarm",
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
        task_difficulty: str = DEFAULT_TASK_DIFFICULTY,
        founders: "list[FounderInjection] | None" = None,
        admixture_commensal_s: float = 0.0,
        admixture_exchange_s: float = 0.0,
        admixture_phase2_blend: float = 0.05,
        admixture_phase2_prob_mul: float = 1.0,
        assortative_temperature: float = float("inf"),
    ):
        self.pop_max = int(pop_max)
        self.rng = rng
        self.task = validate_task(task)
        # Environment shaping (only used when task=l2v2_ctrl). Resolve the
        # preset name *now* so any typo blows up at colony launch, not
        # asynchronously inside the sim loop.
        self.task_difficulty = str(task_difficulty)
        self._difficulty_weights = difficulty_weights(self.task_difficulty)
        # SPEC_L2_V3.0 §1.5 — derive which logic modes can ever be sampled
        # in this difficulty preset.  Specialist dishes (and_only / not_only)
        # have p_mode_and = 1.0 or 0.0 respectively, so one of the buffers
        # is permanently empty.  We use these flags to gate _fitness_defined
        # and _logic_fitness_slot — without them an entire specialist
        # population is forever "fitness undefined", which silently breaks
        # sigma annealing and elite-victim selection (see ERRATA v3.1).
        p_and = float(self._difficulty_weights.get("p_mode_and", 0.5))
        self._needs_and_samples: bool = p_and > 0.0
        self._needs_not_samples: bool = p_and < 1.0
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
        # SPEC_L2_V3.5c — hard-row only ring buffers (per agent).
        # _acc_and_hard_buf[slot, k] = 1 if the slot was correct on the k-th most
        # recent (1,1)-AND window it personally observed, else 0.  _hard_n is
        # min(samples_seen, LOGIC_HARD_HISTORY).  _hard_idx is the next slot to
        # overwrite (FIFO).  Same layout for NOT (a=0).
        self._acc_and_hard_buf = np.zeros((self.pop_max, LOGIC_HARD_HISTORY), dtype=np.int8)
        self._acc_not_hard_buf = np.zeros((self.pop_max, LOGIC_HARD_HISTORY), dtype=np.int8)
        self._acc_and_hard_n = np.zeros(self.pop_max, dtype=np.int32)
        self._acc_not_hard_n = np.zeros(self.pop_max, dtype=np.int32)
        self._acc_and_hard_idx = np.zeros(self.pop_max, dtype=np.int32)
        self._acc_not_hard_idx = np.zeros(self.pop_max, dtype=np.int32)
        self._last_oracle = None  # OracleSample | None
        self._last_consensus_bit: int | None = None
        self._last_consensus_acc: float = 0.0
        self._last_acc_and_pop: float = 0.0
        self._last_acc_not_pop: float = 0.0
        self._last_both_pass_pct: float = 0.0
        self._last_logic_diversity: float = 0.0

        # Population-level row-specific accuracy ring buffers (one slot per row).
        # Each window where row r fires contributes (n_correct_agents, n_alive_agents);
        # the ring buffer keeps the last ROW_HISTORY *firings of that row* (not
        # all windows) so each row gets ~equal smoothing regardless of weight.
        self._row_buf_correct = np.zeros((N_ROW_BUCKETS, ROW_HISTORY), dtype=np.int32)
        self._row_buf_total = np.zeros((N_ROW_BUCKETS, ROW_HISTORY), dtype=np.int32)
        self._row_buf_idx = np.zeros(N_ROW_BUCKETS, dtype=np.int32)
        self._row_buf_filled = np.zeros(N_ROW_BUCKETS, dtype=bool)
        self._last_row_acc: list[float] = [0.0] * N_ROW_BUCKETS
        # SPEC_L2_V3.5b — parallel buffers that only count on-niche voters
        # (AND/dual experts on AND windows, NOT/dual experts on NOT windows).
        # Same shape / index as the legacy buffers; the row index function is
        # shared.  When voters_swarm == 0 the window is silently skipped.
        self._row_buf_correct_swarm = np.zeros(
            (N_ROW_BUCKETS, ROW_HISTORY), dtype=np.int32
        )
        self._row_buf_total_swarm = np.zeros(
            (N_ROW_BUCKETS, ROW_HISTORY), dtype=np.int32
        )

        # SPEC_L2_V3.4 — 3-phase admixture protocol.  Replaces the v3.0
        # "single window with HGT boost" model that turned out to be
        # ecologically backwards: bursting HGT *immediately* on contact
        # means the dominant strain (typically the one with larger weight
        # magnitudes) sweeps the population in seconds, eliminating the
        # competing trait before recombination can produce dual-logic
        # hybrids.  v3.4 mirrors the actual sequence in microbial
        # admixture experiments (e.g. Lenski two-strain coexistence,
        # Synechococcus / Prochlorococcus oceanic mixing):
        #
        #   Phase 1 — Commensal (0 .. commensal_s):
        #       HGT *off entirely*.  Both founder strains share the same
        #       dish but their gene pools stay isolated.  Selection acts
        #       within each strain on its own metabolic regime; both
        #       strains learn to survive in the new mixed environment.
        #
        #   Phase 2 — Controlled exchange (commensal_s .. commensal_s + exchange_s):
        #       HGT enabled but with a *small* blend ratio (default 0.05
        #       vs. baseline 0.30).  Models bacterial conjugation /
        #       transformation: each event transfers a few genes, not a
        #       weighted average of two whole genomes.  Strong-magnitude
        #       weights from one strain no longer dilute their counter-
        #       parts on the other side; small per-event transfers give
        #       evolution many tries to find useful crossovers.
        #
        #   Phase 3 — Restored (t ≥ commensal_s + exchange_s):
        #       HGT returns to the colony's baseline blend / prob.
        #       The protocol's "first contact" period is over; the
        #       population behaves like an ordinary colony from here.
        #
        # See docs/SPEC_L2_V3.0_admixture.md §V3.4 + project-summary §5.12
        # for the rationale and the diagnostic that motivated the change.
        self.admixture_commensal_s = float(max(0.0, admixture_commensal_s))
        self.admixture_exchange_s = float(max(0.0, admixture_exchange_s))
        self.admixture_phase2_blend = float(
            min(1.0, max(0.0, admixture_phase2_blend))
        )
        self.admixture_phase2_prob_mul = float(max(0.0, admixture_phase2_prob_mul))
        self._t_sim_seconds = 0.0

        # SPEC_L2_V3.5 — assortative HGT (prezygotic isolation by niche).
        # ∞ → bit-identical to v3.4 ("richest neighbour wins"); finite values
        # weight donor sampling by exp(-|Δniche|/T) so AND-experts preferentially
        # share genes with AND-experts and NOT with NOT.  See docs/SPEC_L2_V3.5.
        self.assortative_temperature = float(assortative_temperature)
        # Swarm-level dual-logic telemetry caches.
        self._last_species_counts: dict[int, int] = {
            SPECIES_NOVICE: 0,
            SPECIES_AND_EXPERT: 0,
            SPECIES_NOT_EXPERT: 0,
            SPECIES_DUAL_EXPERT: 0,
        }
        self._last_acc_and_swarm: float = 0.0
        self._last_acc_not_swarm: float = 0.0
        self._last_consensus_acc_swarm: float = 0.0
        self._last_consensus_bit_swarm: int | None = None
        self._last_consensus_voters_swarm: int = 0
        self._last_colony_dual_acc: float = 0.0

        if self.slime.enabled:
            init_xy = random_positions(self.rng, self.pop_max, self.slime.grid_size)
            self.positions[...] = init_xy

        # SPEC_L2_V3.0 — admixture: if founders given, draw from those strains
        # for the initial slots; remaining slots fall back to random init so the
        # population stays evolvable even at a low fraction sum.
        if founders:
            self._spawn_initial_with_founders(n0, founders)
        else:
            for i in range(n0):
                self.spawn_initial_slot(i)

    def _init_weights_for_task(self) -> np.ndarray:
        """Sample one fresh weight vector with the task-appropriate prior.

        L1                  : Uniform(-3, 3)        — SPEC §1.1.
        L2v2 (default/mixed): Uniform(-0.5, +1.5)   — SPEC_L2_V2.0 §3.1
                              with v2.5 prebiotic-stage offset
                              (E[w]=+0.5; ~25% negative).  Founders spike
                              enough to bootstrap evolution.
        L2v2 (not_only)     : Anti-follower structural seed — ERRATA v3.3.
                              Hidden layer split into A-detectors (suppress
                              output when A high) and S-tonic drivers
                              (drive output when S high).  Each agent draws
                              independent noise so selection still has
                              variation to operate on.  See the
                              L2V2_NOT_SEED_* comment block above.
        """
        if self.task == TASK_L2V2:
            if self.task_difficulty == DIFFICULTY_ANTI_FOLLOWER_SEED:
                return self._init_weights_l2v2_not_seed()
            return self.rng.uniform(
                L2V2_WEIGHT_INIT_LOW, L2V2_WEIGHT_INIT_HIGH, size=N_WEIGHTS
            )
        return self.rng.uniform(-3.0, 3.0, size=N_WEIGHTS)

    def _init_weights_l2v2_not_seed(self) -> np.ndarray:
        """ERRATA v3.3 (Path D1) — anti-follower structural seed founder.

        Lays down a NOT-prone micro-circuit at birth instead of asking
        random noise to discover one.  Layout (220-vector unpacked into
        w_ih (10×20) and w_ho (20×1)):

            input rows   0..3  = A channels   (N_CH_A=4)
            input rows   4..7  = B channels   (N_CH_B=4)
            input rows   8..9  = S channels   (N_CH_S=2)
            hidden cols  0..H_A_DET-1   = A-detectors
            hidden cols  H_A_DET..19    = S-tonic drivers

        Strong-positive bands wire detectors/drivers; strong-negative band
        wires A-detector→output (suppression); modest-positive band wires
        S-tonic→output (drive).  Everything else gets symmetric small
        noise so mutation has a discoverable substrate (cf. AND-comorbid
        circuits, NOT-on-B variants).
        """
        rng = self.rng
        w = np.empty(N_WEIGHTS, dtype=np.float64)
        w_ih = w[: N_INPUT * N_HIDDEN].reshape(N_INPUT, N_HIDDEN)
        w_ho = w[N_INPUT * N_HIDDEN :].reshape(N_HIDDEN, N_OUTPUT)

        # 1) symmetric small-noise background everywhere
        w_ih[:] = rng.uniform(
            L2V2_NOT_SEED_NOISE_LOW, L2V2_NOT_SEED_NOISE_HIGH, size=w_ih.shape
        )
        w_ho[:] = rng.uniform(
            L2V2_NOT_SEED_NOISE_LOW, L2V2_NOT_SEED_NOISE_HIGH, size=w_ho.shape
        )

        # 2) channel index ranges for A / B / S in the input vector
        a_lo, a_hi = 0, N_CH_A
        s_lo, s_hi = N_CH_A + N_CH_B, N_INPUT
        h_a_det = L2V2_NOT_SEED_H_A_DET

        # 3) A-detectors (hidden cols 0..h_a_det-1)
        #    A→hidden: strong positive  — track A channel
        w_ih[a_lo:a_hi, :h_a_det] = rng.uniform(
            L2V2_NOT_SEED_STRONG_POS_LOW,
            L2V2_NOT_SEED_STRONG_POS_HIGH,
            size=(N_CH_A, h_a_det),
        )
        #    hidden→output: strong NEGATIVE — A high suppresses output
        w_ho[:h_a_det, 0] = rng.uniform(
            L2V2_NOT_SEED_STRONG_NEG_LOW,
            L2V2_NOT_SEED_STRONG_NEG_HIGH,
            size=h_a_det,
        )

        # 4) S-tonic drivers (hidden cols h_a_det..N_HIDDEN-1)
        #    S→hidden: positive — S=80Hz keeps these hidden cells firing
        w_ih[s_lo:s_hi, h_a_det:N_HIDDEN] = rng.uniform(
            L2V2_NOT_SEED_TONIC_POS_LOW,
            L2V2_NOT_SEED_TONIC_POS_HIGH,
            size=(N_CH_S, N_HIDDEN - h_a_det),
        )
        #    hidden→output: positive — drive output unless suppressed
        w_ho[h_a_det:N_HIDDEN, 0] = rng.uniform(
            L2V2_NOT_SEED_OUT_TONIC_LOW,
            L2V2_NOT_SEED_OUT_TONIC_HIGH,
            size=N_HIDDEN - h_a_det,
        )
        return w

    def _reset_logic_buffers_slot(self, slot: int) -> None:
        self._acc_and_buf[slot].fill(0)
        self._acc_not_buf[slot].fill(0)
        self._acc_and_n[slot] = 0
        self._acc_not_n[slot] = 0
        self._acc_and_hits[slot] = 0
        self._acc_not_hits[slot] = 0
        self._acc_and_hard_buf[slot].fill(0)
        self._acc_not_hard_buf[slot].fill(0)
        self._acc_and_hard_n[slot] = 0
        self._acc_not_hard_n[slot] = 0
        self._acc_and_hard_idx[slot] = 0
        self._acc_not_hard_idx[slot] = 0

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

    def _spawn_initial_with_founders(
        self,
        n_initial: int,
        founders: list[FounderInjection],
    ) -> None:
        """SPEC_L2_V3.0 §1.3 — admixture initial spawn.

        Each founder claims ``round(fraction × n_initial)`` slots, filled by
        sampling-with-replacement from its weight pool.  Any leftover slots
        (from rounding or sum(fraction) < 1.0) fall back to task-default
        random init — this keeps founder evolvability if the user only mixes
        a small fraction of saved strains into a fresh sea.
        """
        # Validate fractions; reject obvious user-side mistakes loudly so the
        # sim doesn't silently launch with zero founders from a typo'd payload.
        for f in founders:
            if f.weights.ndim != 2 or f.weights.shape[1] != N_WEIGHTS:
                raise ValueError(
                    f"founder {f.label!r}: weights must be (K, {N_WEIGHTS}), got {f.weights.shape}"
                )
            if f.weights.shape[0] == 0:
                raise ValueError(f"founder {f.label!r} has zero agents")
            if not 0.0 <= f.fraction <= 1.0:
                raise ValueError(f"founder {f.label!r} fraction {f.fraction} not in [0, 1]")
        total = sum(f.fraction for f in founders)
        if total > 1.0 + 1e-9:
            raise ValueError(f"founder fractions sum to {total:.3f} > 1.0")

        # Allocate slot counts by integer share of n_initial.  We round so the
        # *actual* counts are predictable for tests and never overflow n_initial.
        counts = [int(round(f.fraction * n_initial)) for f in founders]
        # Trim if rounding pushed us past n_initial.
        while sum(counts) > n_initial:
            j = int(np.argmax(counts))
            counts[j] -= 1

        slot = 0
        for f, k in zip(founders, counts):
            if k <= 0:
                continue
            picks = self.rng.integers(0, f.weights.shape[0], size=k)
            for w in f.weights[picks]:
                self.spawn_initial_slot(slot)
                # Overwrite the freshly-randomised weights with the founder's.
                # Everything else (credit=C_INIT, fresh membrane, empty buffers)
                # is correct as-is — admixture deliberately discards the source
                # culture's bookkeeping (it's a "spore state": just the genome).
                self.weights[slot] = np.asarray(w, dtype=np.float64).copy()
                slot += 1

        # Fill remaining slots with task-default random init.
        while slot < n_initial:
            self.spawn_initial_slot(slot)
            slot += 1

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
            # In L2v2 we require samples in *every* mode the environment can
            # produce.  For mixed dishes that means BOTH buffers (otherwise a
            # brand-new agent that has only seen AND windows would be ranked
            # as if it could also do NOT).  For specialist dishes
            # (and_only / not_only) one buffer is permanently empty by design,
            # so we only require the buffer for the active mode — without
            # this carve-out an entire specialist population stays
            # "fitness undefined" forever, freezing sigma at SIGMA_BASE and
            # degrading elite-victim selection to credit-only.  ERRATA v3.1.
            and_ok = (not self._needs_and_samples) or int(self._acc_and_n[slot]) > 0
            not_ok = (not self._needs_not_samples) or int(self._acc_not_n[slot]) > 0
            return bool(and_ok and not_ok)
        return int(self._hc[slot]) >= N_HISTORY

    def _logic_acc_slot(self, slot: int, mode: int) -> float:
        if mode == MODE_AND:
            n = int(self._acc_and_n[slot])
            return (int(self._acc_and_hits[slot]) / n) if n > 0 else 0.0
        n = int(self._acc_not_n[slot])
        return (int(self._acc_not_hits[slot]) / n) if n > 0 else 0.0

    def _logic_fitness_slot(self, slot: int) -> float:
        """Mean accuracy across the modes this dish actually samples, in [0, 1].

        Mixed dishes average AND and NOT (the historical behaviour).
        Specialist dishes (and_only / not_only) report just the active
        mode's accuracy — otherwise a perfect NOT-学家 caps at fitness 0.5,
        which compresses sigma annealing's dynamic range and makes the
        cultivated strain look weaker than it is in saved metadata.
        """
        if self._needs_and_samples and self._needs_not_samples:
            return 0.5 * (
                self._logic_acc_slot(slot, MODE_AND)
                + self._logic_acc_slot(slot, MODE_NOT)
            )
        if self._needs_and_samples:
            return self._logic_acc_slot(slot, MODE_AND)
        if self._needs_not_samples:
            return self._logic_acc_slot(slot, MODE_NOT)
        return 0.0

    def _record_row_window(
        self,
        mode: int,
        bit_a: int,
        bit_b: int,
        n_correct: int,
        n_total: int,
        n_correct_swarm: int = 0,
        n_total_swarm: int = 0,
    ) -> None:
        """Population-level: record one window's (n_correct, n_total) on row (mode, a, b).

        This is the data source for the "1∧1 真学会率" / "NOT 0 真学会率"
        dashboard widgets — the only way to tell genuine learning apart
        from the silent attractor.

        SPEC_L2_V3.5b: ``n_*_swarm`` mirrors the same row but counts only
        on-niche voters (AND/dual experts on AND windows, NOT/dual on NOT).
        Both buffers advance the same ring index so they stay aligned.
        """
        if n_total <= 0:
            return
        b = _row_bucket(mode, bit_a, bit_b)
        idx = int(self._row_buf_idx[b])
        self._row_buf_correct[b, idx] = int(n_correct)
        self._row_buf_total[b, idx] = int(n_total)
        self._row_buf_correct_swarm[b, idx] = int(max(0, n_correct_swarm))
        self._row_buf_total_swarm[b, idx] = int(max(0, n_total_swarm))
        self._row_buf_idx[b] = (idx + 1) % ROW_HISTORY
        if idx + 1 >= ROW_HISTORY:
            self._row_buf_filled[b] = True

    def _row_acc(self, mode: int, bit_a: int, bit_b: int) -> float:
        """Mean per-window agent-fraction-correct on row (mode, a, b) over its ring."""
        b = _row_bucket(mode, bit_a, bit_b)
        if self._row_buf_filled[b]:
            tot = int(self._row_buf_total[b].sum())
            cor = int(self._row_buf_correct[b].sum())
        else:
            n = int(self._row_buf_idx[b])
            if n == 0:
                return 0.0
            tot = int(self._row_buf_total[b, :n].sum())
            cor = int(self._row_buf_correct[b, :n].sum())
        return (cor / tot) if tot > 0 else 0.0

    def _row_acc_swarm(self, mode: int, bit_a: int, bit_b: int) -> float:
        """Same as :meth:`_row_acc` but only on-niche voters contribute.

        Returns 0.0 when no on-niche voter has answered this row yet (i.e.
        the relevant species hasn't emerged).  Frontends should treat that
        as 'no data' rather than 'failed'.
        """
        b = _row_bucket(mode, bit_a, bit_b)
        if self._row_buf_filled[b]:
            tot = int(self._row_buf_total_swarm[b].sum())
            cor = int(self._row_buf_correct_swarm[b].sum())
        else:
            n = int(self._row_buf_idx[b])
            if n == 0:
                return 0.0
            tot = int(self._row_buf_total_swarm[b, :n].sum())
            cor = int(self._row_buf_correct_swarm[b, :n].sum())
        return (cor / tot) if tot > 0 else 0.0

    def _row_n_samples_swarm(self, mode: int, bit_a: int, bit_b: int) -> int:
        """How many on-niche voter samples are in the ring for row (mode, a, b)."""
        b = _row_bucket(mode, bit_a, bit_b)
        if self._row_buf_filled[b]:
            return int(self._row_buf_total_swarm[b].sum())
        n = int(self._row_buf_idx[b])
        if n == 0:
            return 0
        return int(self._row_buf_total_swarm[b, :n].sum())

    def _row_n_samples(self, mode: int, bit_a: int, bit_b: int) -> int:
        """How many windows of row (mode, a, b) are in the ring (for UI confidence display)."""
        b = _row_bucket(mode, bit_a, bit_b)
        return int(ROW_HISTORY if self._row_buf_filled[b] else self._row_buf_idx[b])

    # ── SPEC_L2_V3.5 — niche / species classification ──────────────────────

    def _niche_slot(self, slot: int) -> float:
        """Per-agent niche scalar in [-1, +1]: acc_AND - acc_NOT.

        + 1 ⇒ pure AND-expert; -1 ⇒ pure NOT-expert; 0 ⇒ generalist (or
        untrained novice — caller should check sample counts via
        ``_species_of_slot`` if it needs to distinguish those).
        """
        return float(
            self._logic_acc_slot(slot, MODE_AND)
            - self._logic_acc_slot(slot, MODE_NOT)
        )

    def _species_of_slot(self, slot: int) -> int:
        """Classify one living agent into the four species quadrants.

        SPEC_L2_V3.5 §2 + SPEC_L2_V3.5c hard-row qualification.

        An agent counts as AND_EXPERT only if **both** hold:
          * General per-mode rolling accuracy ≥ NICHE_SPECIALIST_THRESHOLD with
            ≥ NICHE_MIN_SAMPLES of evidence (the original v3.5 bar), AND
          * Hard-row rolling accuracy on (1,1)-AND windows ≥
            NICHE_HARD_THRESHOLD with ≥ NICHE_HARD_MIN_SAMPLES of those
            specific windows (the v3.5c "actually fires when needed" bar).

        Same dual-bar applies to NOT_EXPERT against the (a=0)-NOT row.

        Rationale: under task_difficulty=balanced the easy target=0 rows
        give a 50 % floor for free, which lets a half-firing agent slip past
        the 0.65 mode bar.  Adding the hard-row bar refuses the free pass —
        the agent must demonstrate it can actually produce '1' on the row
        that demands it, not just stay silent on the easy zeros.
        """
        n_and = int(self._acc_and_n[slot])
        n_not = int(self._acc_not_n[slot])
        n_and_hard = int(self._acc_and_hard_n[slot])
        n_not_hard = int(self._acc_not_hard_n[slot])
        and_ok = (
            n_and >= NICHE_MIN_SAMPLES
            and self._logic_acc_slot(slot, MODE_AND) >= NICHE_SPECIALIST_THRESHOLD
            and n_and_hard >= NICHE_HARD_MIN_SAMPLES
            and self._logic_hard_acc_slot(slot, MODE_AND) >= NICHE_HARD_THRESHOLD
        )
        not_ok = (
            n_not >= NICHE_MIN_SAMPLES
            and self._logic_acc_slot(slot, MODE_NOT) >= NICHE_SPECIALIST_THRESHOLD
            and n_not_hard >= NICHE_HARD_MIN_SAMPLES
            and self._logic_hard_acc_slot(slot, MODE_NOT) >= NICHE_HARD_THRESHOLD
        )
        if and_ok and not_ok:
            return SPECIES_DUAL_EXPERT
        if and_ok:
            return SPECIES_AND_EXPERT
        if not_ok:
            return SPECIES_NOT_EXPERT
        return SPECIES_NOVICE

    def _niche_array_living(self, alive_idx: np.ndarray) -> np.ndarray:
        """Vector of niche scalars (acc_AND - acc_NOT) aligned to `alive_idx`."""
        out = np.zeros(alive_idx.size, dtype=np.float64)
        for j, s in enumerate(alive_idx.tolist()):
            out[j] = self._niche_slot(int(s))
        return out

    def _record_logic_window(
        self, slot: int, mode: int, correct: bool, target_bit: int = 0
    ) -> None:
        """Record one window's (slot, mode, correct) into the rolling buffers.

        SPEC_L2_V3.5c — also records a separate "hard-row" buffer when
        ``target_bit == 1`` (the rows that require the agent to actually fire:
        (1,1)-AND and (a=0)-NOT).  These hard buffers are used by
        :meth:`_species_of_slot` to filter out silent-pretender specialists.
        The general per-mode buffer is unchanged.
        """
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

        if int(target_bit) == 1:
            if mode == MODE_AND:
                hard_idx = int(self._acc_and_hard_idx[slot])
                self._acc_and_hard_buf[slot, hard_idx] = 1 if correct else 0
                self._acc_and_hard_idx[slot] = (hard_idx + 1) % LOGIC_HARD_HISTORY
                if int(self._acc_and_hard_n[slot]) < LOGIC_HARD_HISTORY:
                    self._acc_and_hard_n[slot] = int(self._acc_and_hard_n[slot]) + 1
            else:
                hard_idx = int(self._acc_not_hard_idx[slot])
                self._acc_not_hard_buf[slot, hard_idx] = 1 if correct else 0
                self._acc_not_hard_idx[slot] = (hard_idx + 1) % LOGIC_HARD_HISTORY
                if int(self._acc_not_hard_n[slot]) < LOGIC_HARD_HISTORY:
                    self._acc_not_hard_n[slot] = int(self._acc_not_hard_n[slot]) + 1

    def _logic_hard_acc_slot(self, slot: int, mode: int) -> float:
        """Per-slot rolling accuracy on the *hard* row only (target_bit=1).

        Returns 0.0 when no hard-row evidence has been observed yet.
        """
        if mode == MODE_AND:
            n = int(self._acc_and_hard_n[slot])
            if n <= 0:
                return 0.0
            return float(self._acc_and_hard_buf[slot, :n].sum()) / float(n)
        n = int(self._acc_not_hard_n[slot])
        if n <= 0:
            return 0.0
        return float(self._acc_not_hard_buf[slot, :n].sum()) / float(n)

    def _logic_hard_n_slot(self, slot: int, mode: int) -> int:
        """Number of hard-row windows observed by this slot in current life."""
        if mode == MODE_AND:
            return int(self._acc_and_hard_n[slot])
        return int(self._acc_not_hard_n[slot])

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
            w = self._difficulty_weights
            oracle = draw_oracle_sample(
                self.rng,
                p_mode_and=w["p_mode_and"],
                p_and_target_one=w["p_and_target_one"],
                p_not_target_one=w["p_not_target_one"],
                # ERRATA v3.2 — specialist dishes pass -BREATH_PER_WINDOW so
                # silent agents stop being free-riders.  Mixed dishes leave
                # this at 0 (default) preserving the v2.2 anti-collapse
                # invariant.  .get() keeps backward compat with any external
                # preset that predates the reward_wrong field.
                reward_wrong=w.get("reward_wrong", 0.0),
            )
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
                self._record_logic_window(slot, oracle.mode, correct, oracle.target_bit)
                if correct:
                    rewards[j] = oracle.reward_correct
                    n_correct += 1
                else:
                    rewards[j] = oracle.reward_wrong
                # ERRATA v2.4 (design "C"): add a small directionally-correct
                # spike-effort bonus on top of the table reward.  This is the
                # gradient that lets near-spike mutations (e.g. 16 Hz on a (1,1)
                # window) pay slightly better than fully-silent ones — turning
                # the platform-cliff fitness landscape into an upward slope.
                # Kept additive (not multiplicative) and applied BEFORE the
                # slime pheromone bonus so it doesn't get amplified into the
                # reward distortion that previously broke the silent attractor.
                rewards[j] += spike_effort_bonus(float(f_outs[j]), oracle.target_bit)
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

            # SPEC_L2_V3.5b — niche-aware consensus.  Only agents whose niche
            # matches the current oracle mode (AND-experts or dual-experts on
            # AND windows; NOT-experts or dual on NOT windows) get a vote.
            # When zero on-niche voters exist (e.g. early evolution, or one
            # specialist niche extinct), we surface (None, 0) so the UI can
            # render "尚无 X 专家" rather than mis-counting silent off-niche
            # voters as "abstaining yes".
            n_correct_swarm = 0
            n_voters_swarm = 0
            spiking_swarm = 0
            if n_alive > 0:
                want_and = (oracle.mode == MODE_AND)
                for j, slot in enumerate(idx.tolist()):
                    sp = self._species_of_slot(int(slot))
                    if sp == SPECIES_DUAL_EXPERT or (
                        want_and and sp == SPECIES_AND_EXPERT
                    ) or (
                        (not want_and) and sp == SPECIES_NOT_EXPERT
                    ):
                        n_voters_swarm += 1
                        if int(out_bits[j]) == oracle.target_bit:
                            n_correct_swarm += 1
                        if int(out_bits[j]) == 1:
                            spiking_swarm += 1
            if n_voters_swarm > 0:
                consensus_bit_swarm = (
                    1 if spiking_swarm >= (n_voters_swarm - spiking_swarm) else 0
                )
                consensus_acc_swarm = float(n_correct_swarm / n_voters_swarm)
            else:
                consensus_bit_swarm = None
                consensus_acc_swarm = 0.0
            self._last_consensus_acc_swarm = consensus_acc_swarm
            self._last_consensus_bit_swarm = consensus_bit_swarm
            self._last_consensus_voters_swarm = int(n_voters_swarm)

            # Population-level per-row accuracy (the "真学会" gauge data).
            # n_correct already counted above; record it against this window's
            # specific (mode, a, b) row.  n_*_swarm threads niche-aware tallies
            # into the parallel ring buffer.
            if n_alive > 0:
                self._record_row_window(
                    oracle.mode, oracle.bit_a, oracle.bit_b,
                    n_correct=n_correct, n_total=n_alive,
                    n_correct_swarm=n_correct_swarm,
                    n_total_swarm=n_voters_swarm,
                )

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
        # SPEC_L2_V3.4 — 3-phase admixture protocol (see __init__ comment).
        # phase==1 → HGT entirely skipped (commensal isolation).
        # phase==2 → HGT enabled with reduced blend (controlled exchange).
        # phase==3 → baseline HGT (post-admixture, normal colony life).
        admix_phase = self._admixture_phase()
        if admix_phase == 1:
            eff_hgt_prob = 0.0
            eff_hgt_blend = 0.0
        elif admix_phase == 2:
            eff_hgt_prob = min(
                1.0, self.slime.hgt_prob * self.admixture_phase2_prob_mul
            )
            eff_hgt_blend = self.admixture_phase2_blend
        else:
            eff_hgt_prob = self.slime.hgt_prob
            eff_hgt_blend = self.slime.hgt_blend
        hgt_count = 0
        hgt_pair_log: list[tuple[int, int]] = []
        if (
            self.slime.enabled
            and self.slime.hgt_enabled
            and n_alive >= 2
            and eff_hgt_prob > 0.0
        ):
            pos_alive = self.positions[idx]
            cred_alive = self.credit[idx].copy()
            # SPEC_L2_V3.5 — only compute the niche vector when assortative HGT
            # is actually requested (finite temperature).  Default ∞ keeps the
            # call signature legacy-equivalent for L1 and v3.4 colonies.
            niche_alive: np.ndarray | None
            if (
                self.task == TASK_L2V2
                and np.isfinite(self.assortative_temperature)
            ):
                niche_alive = self._niche_array_living(idx)
            else:
                niche_alive = None
            pairs = hgt_pairs(
                self.rng,
                pos_alive,
                cred_alive,
                self.slime.grid_size,
                self.slime.hgt_radius,
                eff_hgt_prob,
                self.slime.hgt_donor_ratio,
                niche=niche_alive,
                assortative_temperature=self.assortative_temperature,
            )
            for r_local, d_local in pairs:
                r_slot = int(idx[r_local])
                d_slot = int(idx[d_local])
                if not self.alive[r_slot] or not self.alive[d_slot]:
                    continue
                if self.credit[r_slot] <= self.slime.hgt_cost:
                    continue
                self.weights[r_slot] = blend_weights(
                    self.weights[r_slot], self.weights[d_slot], eff_hgt_blend
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
                # SPEC_L2_V3.5 — species classification + niche-aware swarm
                # metrics.  After the v3.4 admixture failure (§5.12) we accept
                # that L2 dual-logic does NOT have to live in a single agent;
                # instead it can live in the colony as a community of two
                # specialists.  These metrics measure exactly that.
                species_counts: dict[int, int] = {
                    SPECIES_NOVICE: 0,
                    SPECIES_AND_EXPERT: 0,
                    SPECIES_NOT_EXPERT: 0,
                    SPECIES_DUAL_EXPERT: 0,
                }
                # Two votes-by-niche means: AND-experts (incl. dual) vote on AND,
                # NOT-experts (incl. dual) vote on NOT.  Generalists / novices do
                # NOT vote — they're not authorities on either question, and
                # including them is what the v3.0 acc_*_pop metric does wrong.
                and_voters_acc: list[float] = []
                not_voters_acc: list[float] = []
                for s_int in alive_idx.tolist():
                    s = int(s_int)
                    sp = self._species_of_slot(s)
                    species_counts[sp] += 1
                    if sp == SPECIES_AND_EXPERT or sp == SPECIES_DUAL_EXPERT:
                        and_voters_acc.append(self._logic_acc_slot(s, MODE_AND))
                    if sp == SPECIES_NOT_EXPERT or sp == SPECIES_DUAL_EXPERT:
                        not_voters_acc.append(self._logic_acc_slot(s, MODE_NOT))
                acc_and_swarm = (
                    float(np.mean(and_voters_acc)) if and_voters_acc else 0.0
                )
                acc_not_swarm = (
                    float(np.mean(not_voters_acc)) if not_voters_acc else 0.0
                )
                # colony_dual_acc — the L2 SPEC_V3.5 success indicator.
                # Defined only when BOTH species have ≥ NICHE_MIN_SAMPLES voters
                # (otherwise the colony is monolithic and one species's metric is
                # vacuously 0; reporting that as 0.5 × acc_specialist would lie).
                if (
                    len(and_voters_acc) >= NICHE_MIN_SAMPLES
                    and len(not_voters_acc) >= NICHE_MIN_SAMPLES
                ):
                    colony_dual_acc = 0.5 * (acc_and_swarm + acc_not_swarm)
                else:
                    colony_dual_acc = 0.0
            else:
                acc_and_pop = 0.0
                acc_not_pop = 0.0
                both_pass_pct = 0.0
                logic_diversity = 0.0
                species_counts = {
                    SPECIES_NOVICE: 0,
                    SPECIES_AND_EXPERT: 0,
                    SPECIES_NOT_EXPERT: 0,
                    SPECIES_DUAL_EXPERT: 0,
                }
                acc_and_swarm = 0.0
                acc_not_swarm = 0.0
                colony_dual_acc = 0.0
            self._last_acc_and_pop = acc_and_pop
            self._last_acc_not_pop = acc_not_pop
            self._last_both_pass_pct = both_pass_pct
            self._last_logic_diversity = logic_diversity
            self._last_species_counts = species_counts
            self._last_acc_and_swarm = acc_and_swarm
            self._last_acc_not_swarm = acc_not_swarm
            self._last_colony_dual_acc = colony_dual_acc

            # Cache row-specific accuracies for telemetry payload.
            # Index order matches _row_bucket(): AND(0,0), AND(0,1), AND(1,0),
            # AND(1,1), NOT(a=0), NOT(a=1).
            self._last_row_acc = [
                self._row_acc(MODE_AND, 0, 0),
                self._row_acc(MODE_AND, 0, 1),
                self._row_acc(MODE_AND, 1, 0),
                self._row_acc(MODE_AND, 1, 1),
                self._row_acc(MODE_NOT, 0, 0),  # b=0 sentinel; NOT bucket only depends on a
                self._row_acc(MODE_NOT, 1, 0),
            ]

        # Advance the population's own t_sim clock — used by the admixture
        # window check above.  Each window is 500 ms by SPEC §3.
        self._t_sim_seconds += 0.5

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
            # SPEC_L2_V3.5 — niche-aware swarm-level dual-logic metrics.
            # acc_*_swarm = per-question-type expert vote (no novices/cross-pollution).
            # colony_dual_acc = (acc_and_swarm + acc_not_swarm)/2, only defined when
            # both species exist in sufficient numbers (else 0).
            # species_counts gives the four-quadrant population breakdown.
            "acc_and_swarm": (
                float(self._last_acc_and_swarm) if self.task == TASK_L2V2 else 0.0
            ),
            "acc_not_swarm": (
                float(self._last_acc_not_swarm) if self.task == TASK_L2V2 else 0.0
            ),
            "colony_dual_acc": (
                float(self._last_colony_dual_acc) if self.task == TASK_L2V2 else 0.0
            ),
            "species_counts": (
                {
                    "novice": int(self._last_species_counts.get(SPECIES_NOVICE, 0)),
                    "and_expert": int(self._last_species_counts.get(SPECIES_AND_EXPERT, 0)),
                    "not_expert": int(self._last_species_counts.get(SPECIES_NOT_EXPERT, 0)),
                    "dual_expert": int(self._last_species_counts.get(SPECIES_DUAL_EXPERT, 0)),
                }
                if self.task == TASK_L2V2
                else None
            ),
            "assortative_temperature": float(self.assortative_temperature),
            # SPEC_L2_V3.5b — niche-aware consensus.  consensus_acc_swarm is
            # the "ask only the matching specialists" version of consensus_acc;
            # consensus_voters_swarm tells the UI how many on-niche voters
            # backed it (0 = no specialists exist yet, treat as 'no data').
            "consensus_acc_swarm": (
                float(self._last_consensus_acc_swarm) if self.task == TASK_L2V2 else 0.0
            ),
            "consensus_bit_swarm": (
                self._last_consensus_bit_swarm if self.task == TASK_L2V2 else None
            ),
            "consensus_voters_swarm": (
                int(self._last_consensus_voters_swarm) if self.task == TASK_L2V2 else 0
            ),
            # Row-specific accuracy & sample-count (the "真学会" telemetry).
            # acc_and_11_pop is the headline gauge — silent agents are pinned at 0
            # here even when混合 acc_and_pop sits at the silent ceiling (75%/50%).
            "acc_and_11_pop": (
                float(self._row_acc(MODE_AND, 1, 1)) if self.task == TASK_L2V2 else 0.0
            ),
            "acc_not_0_pop": (
                float(self._row_acc(MODE_NOT, 0, 0)) if self.task == TASK_L2V2 else 0.0
            ),
            # SPEC_L2_V3.5b — same headline rows but only on-niche voters
            # contribute, so the silent-majority dilution disappears.
            "acc_and_11_swarm": (
                float(self._row_acc_swarm(MODE_AND, 1, 1)) if self.task == TASK_L2V2 else 0.0
            ),
            "acc_not_0_swarm": (
                float(self._row_acc_swarm(MODE_NOT, 0, 0)) if self.task == TASK_L2V2 else 0.0
            ),
            # All 6 rows + sample counts, for the detailed truth-table widget.
            "row_acc": (
                {
                    "and_00": float(self._row_acc(MODE_AND, 0, 0)),
                    "and_01": float(self._row_acc(MODE_AND, 0, 1)),
                    "and_10": float(self._row_acc(MODE_AND, 1, 0)),
                    "and_11": float(self._row_acc(MODE_AND, 1, 1)),
                    "not_a0": float(self._row_acc(MODE_NOT, 0, 0)),
                    "not_a1": float(self._row_acc(MODE_NOT, 1, 0)),
                }
                if self.task == TASK_L2V2
                else None
            ),
            "row_n": (
                {
                    "and_00": int(self._row_n_samples(MODE_AND, 0, 0)),
                    "and_01": int(self._row_n_samples(MODE_AND, 0, 1)),
                    "and_10": int(self._row_n_samples(MODE_AND, 1, 0)),
                    "and_11": int(self._row_n_samples(MODE_AND, 1, 1)),
                    "not_a0": int(self._row_n_samples(MODE_NOT, 0, 0)),
                    "not_a1": int(self._row_n_samples(MODE_NOT, 1, 0)),
                }
                if self.task == TASK_L2V2
                else None
            ),
            # SPEC_L2_V3.5b — niche-aware row accuracies + voter counts.
            # row_acc_swarm["and_11"] is the *real* headline ('AND-experts get
            # 1∧1 = 1') — only the matching species' votes count, so silent
            # off-niche agents stop dragging the average toward 50%.
            "row_acc_swarm": (
                {
                    "and_00": float(self._row_acc_swarm(MODE_AND, 0, 0)),
                    "and_01": float(self._row_acc_swarm(MODE_AND, 0, 1)),
                    "and_10": float(self._row_acc_swarm(MODE_AND, 1, 0)),
                    "and_11": float(self._row_acc_swarm(MODE_AND, 1, 1)),
                    "not_a0": float(self._row_acc_swarm(MODE_NOT, 0, 0)),
                    "not_a1": float(self._row_acc_swarm(MODE_NOT, 1, 0)),
                }
                if self.task == TASK_L2V2
                else None
            ),
            "row_n_swarm": (
                {
                    "and_00": int(self._row_n_samples_swarm(MODE_AND, 0, 0)),
                    "and_01": int(self._row_n_samples_swarm(MODE_AND, 0, 1)),
                    "and_10": int(self._row_n_samples_swarm(MODE_AND, 1, 0)),
                    "and_11": int(self._row_n_samples_swarm(MODE_AND, 1, 1)),
                    "not_a0": int(self._row_n_samples_swarm(MODE_NOT, 0, 0)),
                    "not_a1": int(self._row_n_samples_swarm(MODE_NOT, 1, 0)),
                }
                if self.task == TASK_L2V2
                else None
            ),
            "task_difficulty": (
                str(self.task_difficulty) if self.task == TASK_L2V2 else None
            ),
            # SPEC_L2_V3.4 — 3-phase admixture telemetry (UI shows phase pill,
            # remaining seconds, current effective HGT params).
            "admixture_phase": int(admix_phase),  # 1 = commensal, 2 = exchange, 3 = restored
            "admixture_active": bool(admix_phase in (1, 2)),
            "admixture_commensal_s": float(self.admixture_commensal_s),
            "admixture_exchange_s": float(self.admixture_exchange_s),
            "admixture_phase2_blend": float(self.admixture_phase2_blend),
            "admixture_phase2_prob_mul": float(self.admixture_phase2_prob_mul),
            "eff_hgt_prob": float(eff_hgt_prob),
            "eff_hgt_blend": float(eff_hgt_blend),
        }

    def _admixture_phase(self) -> int:
        """SPEC_L2_V3.4 — return current admixture phase (1, 2, or 3).

        - 1 = Commensal (HGT off): t < commensal_s
        - 2 = Controlled exchange (low blend): commensal_s ≤ t < commensal_s + exchange_s
        - 3 = Restored (baseline HGT): t ≥ commensal_s + exchange_s, OR
              the protocol was never activated (commensal_s == exchange_s == 0).
        """
        if self.admixture_commensal_s <= 0.0 and self.admixture_exchange_s <= 0.0:
            return 3  # protocol disabled — colony runs on baseline HGT from t=0
        t = self._t_sim_seconds
        if t < self.admixture_commensal_s:
            return 1
        if t < self.admixture_commensal_s + self.admixture_exchange_s:
            return 2
        return 3

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

    def top_k_slots_by_niche(
        self,
        k: int,
        niche: str,
    ) -> np.ndarray:
        """SPEC_L2_V3.5b — pick top-K agents that match a niche.

        ``niche`` ∈ {"and_expert", "not_expert", "dual_expert", "any_expert"}.

        Filtering is by :meth:`_species_of_slot` (uses rolling per-mode
        accuracy + min-sample threshold).  Within the matching set agents
        are ranked by per-mode accuracy of the relevant mode (rather than
        the global fitness average), so e.g. asking for ``and_expert``
        returns the agents that are *most* accurate on AND, not the ones
        with the highest mean(acc_AND, acc_NOT).

        Returns an empty array when no agent matches the niche — callers
        (Use page, 6-题考试) should fall back to ``top_k_slots`` and warn
        the user that the colony lacks that specialist.
        """
        idx = self.living_indices()
        if idx.size == 0:
            return idx

        # SPEC_L2_V3.5c — rank by *hard-row* accuracy (the row that proves the
        # agent can actually fire on demand), with the general per-mode acc as
        # tiebreaker.  This pushes silent-pretenders (which now don't even pass
        # _species_of_slot's hard-row bar, but might if thresholds drift) to
        # the bottom and surfaces the most useful experts to the Use page.
        keep = []
        scores = []
        for s in idx.tolist():
            slot = int(s)
            sp = self._species_of_slot(slot)
            if niche == "and_expert":
                ok = sp in (SPECIES_AND_EXPERT, SPECIES_DUAL_EXPERT)
                rank = (
                    0.7 * self._logic_hard_acc_slot(slot, MODE_AND)
                    + 0.3 * self._logic_acc_slot(slot, MODE_AND)
                )
            elif niche == "not_expert":
                ok = sp in (SPECIES_NOT_EXPERT, SPECIES_DUAL_EXPERT)
                rank = (
                    0.7 * self._logic_hard_acc_slot(slot, MODE_NOT)
                    + 0.3 * self._logic_acc_slot(slot, MODE_NOT)
                )
            elif niche == "dual_expert":
                ok = sp == SPECIES_DUAL_EXPERT
                rank = 0.5 * (
                    0.7 * self._logic_hard_acc_slot(slot, MODE_AND)
                    + 0.3 * self._logic_acc_slot(slot, MODE_AND)
                    + 0.7 * self._logic_hard_acc_slot(slot, MODE_NOT)
                    + 0.3 * self._logic_acc_slot(slot, MODE_NOT)
                )
            elif niche == "any_expert":
                ok = sp != SPECIES_NOVICE
                rank = self._fitness_slot(slot) if self._fitness_defined(slot) else -1.0
            else:
                raise ValueError(f"unknown niche: {niche}")
            if ok:
                keep.append(slot)
                scores.append(float(rank))
        if not keep:
            return np.empty(0, dtype=idx.dtype)
        keep_arr = np.asarray(keep, dtype=idx.dtype)
        order = np.argsort(np.asarray(scores), kind="stable")[::-1]
        pick = order[: min(k, order.size)]
        return keep_arr[pick]

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
