"""SPEC_L2_V3.5 — speciation, niche, assortative HGT, swarm-level dual logic.

These tests cover the four pieces introduced in v3.5:

1. ``hgt_pairs`` with a niche vector + finite temperature picks niche-similar
   donors with strictly higher probability than niche-distant ones.
2. ``Population._species_of_slot`` quadrant-classifies an agent based on its
   per-mode rolling accuracy *and* sample sufficiency.
3. ``Population.step_window`` exposes ``species_counts`` /
   ``acc_*_swarm`` / ``colony_dual_acc`` in its info dict.
4. ``colony_dual_acc`` is non-zero ONLY when both species are populated above
   ``NICHE_MIN_SAMPLES`` voters (otherwise a 100%-AND colony would fake L2).
"""

from __future__ import annotations

import math

import numpy as np

from archaea.population import (
    LOGIC_HARD_HISTORY,
    LOGIC_HISTORY,
    NICHE_HARD_MIN_SAMPLES,
    NICHE_HARD_THRESHOLD,
    NICHE_MIN_SAMPLES,
    NICHE_SPECIALIST_THRESHOLD,
    SPECIES_AND_EXPERT,
    SPECIES_DUAL_EXPERT,
    SPECIES_NOT_EXPERT,
    SPECIES_NOVICE,
    Population,
)
from archaea.oracle import MODE_AND, MODE_NOT
from archaea.slime import SlimeConfig, hgt_pairs
from archaea.task import TASK_L2V2


# ── 1. Assortative HGT primitive (slime.hgt_pairs) ──────────────────────────


def test_hgt_pairs_assortative_prefers_same_niche_donors() -> None:
    """With finite T, a recipient with niche=+1 should pick donors with niche≈+1
    far more often than donors with niche=-1, even when both are richer."""
    rng = np.random.default_rng(0)
    grid = 8
    # Recipient at (0,0) niche=+1 (AND-expert).  All donors at (0,1) — same
    # cell distance, so spatial filter is moot — equal credit.  Half are
    # AND-expert (niche≈+1), half NOT-expert (niche≈-1).
    n = 1 + 40
    positions = np.zeros((n, 2), dtype=np.int32)
    positions[1:21] = (1, 0)   # 20 AND-expert donors
    positions[21:] = (0, 1)    # 20 NOT-expert donors
    credits = np.full(n, 10.0, dtype=np.float64)
    credits[0] = 1.0  # recipient is poor → eligible for HGT
    niche = np.zeros(n, dtype=np.float64)
    niche[0] = +1.0
    niche[1:21] = +1.0
    niche[21:] = -1.0

    same = 0
    diff = 0
    for trial_seed in range(80):
        rng_t = np.random.default_rng(trial_seed)
        pairs = hgt_pairs(
            rng_t, positions, credits, grid_size=grid, radius=2,
            prob=1.0, donor_ratio=2.0,
            niche=niche, assortative_temperature=0.1,
        )
        # Recipient = slot 0 should always fire when prob=1 and donors exist.
        assert any(r == 0 for r, _ in pairs), pairs
        for r, d in pairs:
            if r != 0:
                continue
            if niche[d] > 0:
                same += 1
            else:
                diff += 1
    # With T=0.1 and Δniche=2, e^(-20) ≈ 2e-9; same-niche should dominate by
    # essentially 100%.  Use a generous bound to keep the test robust.
    assert same >= 70, (same, diff)
    assert diff <= 5, (same, diff)


def test_hgt_pairs_legacy_when_niche_omitted_or_inf_temperature() -> None:
    """Default call (no niche) must be bit-identical to v3.4 (richest wins)."""
    rng = np.random.default_rng(0)
    grid = 4
    positions = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    credits = np.array([1.0, 100.0, 5.0], dtype=np.float64)
    pairs_legacy = hgt_pairs(
        np.random.default_rng(123), positions, credits,
        grid_size=grid, radius=1, prob=1.0, donor_ratio=2.0,
    )
    # Recipient 0 is poor; donor 1 is *much* richer than donor 2 → must win.
    assert (0, 1) in pairs_legacy, pairs_legacy

    niche = np.array([+1.0, -1.0, +1.0])  # same niche as donor 2, opposite to donor 1
    pairs_inf = hgt_pairs(
        np.random.default_rng(123), positions, credits,
        grid_size=grid, radius=1, prob=1.0, donor_ratio=2.0,
        niche=niche, assortative_temperature=math.inf,
    )
    assert pairs_inf == pairs_legacy, (pairs_inf, pairs_legacy)


def test_hgt_pairs_assortative_can_pick_poorer_same_niche_donor() -> None:
    """Strict speciation (T → 0) must prefer a niche-twin even if it's poorer."""
    rng = np.random.default_rng(0)
    grid = 4
    positions = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int32)
    # Donor 1 is much richer but opposite niche; donor 2 same niche, only 2x.
    credits = np.array([1.0, 100.0, 2.5], dtype=np.float64)
    niche = np.array([+1.0, -1.0, +1.0])
    same_picks = 0
    for s in range(40):
        pairs = hgt_pairs(
            np.random.default_rng(s), positions, credits,
            grid_size=grid, radius=1, prob=1.0, donor_ratio=2.0,
            niche=niche, assortative_temperature=0.05,
        )
        for r, d in pairs:
            if r == 0 and d == 2:
                same_picks += 1
    assert same_picks >= 35, same_picks  # ≥ 87% should pick the niche-twin


# ── 2. Species classification ───────────────────────────────────────────────


def _make_pop(n: int = 10, seed: int = 9) -> Population:
    rng = np.random.default_rng(seed)
    return Population(
        pop_max=n, rng=rng, n_initial=n,
        task=TASK_L2V2, task_difficulty="balanced",
    )


def _force_history(
    pop: Population, slot: int, *, mode: int,
    n_correct: int, n_total: int,
    hard_n_correct: int | None = None, hard_n_total: int | None = None,
) -> None:
    """Inject a synthetic per-mode rolling history into one slot.

    Bypasses the LIF integration so we can unit-test classification thresholds
    without running thousands of windows.

    SPEC_L2_V3.5c — also fills the per-slot hard-row buffer.  By default the
    hard-row sees the same fraction-correct as the general buffer (so an agent
    that's "18/20 on AND" also gets "9/10 on (1,1)-AND") which keeps existing
    tests valid under the new hard-row requirement.  Pass explicit
    ``hard_n_correct`` / ``hard_n_total`` to test the boundary.
    """
    from archaea.population import LOGIC_HARD_HISTORY  # local import to avoid cycle in fixtures
    assert n_correct <= n_total <= LOGIC_HISTORY
    if mode == MODE_AND:
        pop._acc_and_buf[slot, :n_total] = 0
        pop._acc_and_buf[slot, :n_correct] = 1
        pop._acc_and_n[slot] = n_total
        pop._acc_and_hits[slot] = n_correct
    else:
        pop._acc_not_buf[slot, :n_total] = 0
        pop._acc_not_buf[slot, :n_correct] = 1
        pop._acc_not_n[slot] = n_total
        pop._acc_not_hits[slot] = n_correct

    if hard_n_total is None:
        hard_n_total = min(LOGIC_HARD_HISTORY, max(0, n_total // 2))
    if hard_n_correct is None:
        if n_total > 0:
            hard_n_correct = int(round(hard_n_total * (n_correct / n_total)))
        else:
            hard_n_correct = 0
    assert 0 <= hard_n_correct <= hard_n_total <= LOGIC_HARD_HISTORY
    if mode == MODE_AND:
        pop._acc_and_hard_buf[slot, :hard_n_total] = 0
        pop._acc_and_hard_buf[slot, :hard_n_correct] = 1
        pop._acc_and_hard_n[slot] = hard_n_total
        pop._acc_and_hard_idx[slot] = hard_n_total % LOGIC_HARD_HISTORY
    else:
        pop._acc_not_hard_buf[slot, :hard_n_total] = 0
        pop._acc_not_hard_buf[slot, :hard_n_correct] = 1
        pop._acc_not_hard_n[slot] = hard_n_total
        pop._acc_not_hard_idx[slot] = hard_n_total % LOGIC_HARD_HISTORY


def test_species_classification_quadrants() -> None:
    pop = _make_pop(n=4)
    # slot 0: AND-expert (high acc_AND ≥ threshold, plenty of samples; no NOT data)
    _force_history(pop, 0, mode=MODE_AND, n_correct=18, n_total=20)
    # slot 1: NOT-expert
    _force_history(pop, 1, mode=MODE_NOT, n_correct=18, n_total=20)
    # slot 2: dual expert
    _force_history(pop, 2, mode=MODE_AND, n_correct=18, n_total=20)
    _force_history(pop, 2, mode=MODE_NOT, n_correct=18, n_total=20)
    # slot 3: novice (high acc but only 3 samples — below NICHE_MIN_SAMPLES)
    _force_history(pop, 3, mode=MODE_AND, n_correct=3, n_total=3)

    assert pop._species_of_slot(0) == SPECIES_AND_EXPERT
    assert pop._species_of_slot(1) == SPECIES_NOT_EXPERT
    assert pop._species_of_slot(2) == SPECIES_DUAL_EXPERT
    assert pop._species_of_slot(3) == SPECIES_NOVICE


def test_species_below_threshold_is_novice() -> None:
    """Plenty of samples but acc just under threshold → novice, not expert."""
    pop = _make_pop(n=2)
    threshold = NICHE_SPECIALIST_THRESHOLD
    n_correct = int(LOGIC_HISTORY * (threshold - 0.1))
    _force_history(pop, 0, mode=MODE_AND, n_correct=n_correct, n_total=LOGIC_HISTORY)
    _force_history(pop, 0, mode=MODE_NOT, n_correct=n_correct, n_total=LOGIC_HISTORY)
    assert pop._species_of_slot(0) == SPECIES_NOVICE


# ── 3. Telemetry exposes species_counts + colony_dual_acc ───────────────────


def test_step_window_emits_species_telemetry() -> None:
    pop = _make_pop(n=6)
    info = pop.step_window()
    assert "species_counts" in info
    counts = info["species_counts"]
    assert counts is not None
    assert set(counts.keys()) == {"novice", "and_expert", "not_expert", "dual_expert"}
    assert sum(counts.values()) == int(pop.n_living())
    # Newborns with empty buffers are all novices.
    assert counts["novice"] == int(pop.n_living())
    assert "acc_and_swarm" in info
    assert "acc_not_swarm" in info
    assert "colony_dual_acc" in info
    assert "assortative_temperature" in info
    # Default population uses ∞ (legacy / no assortative bias).
    assert math.isinf(info["assortative_temperature"])


def test_colony_dual_acc_requires_both_species() -> None:
    """If only AND-experts exist, colony_dual_acc must be 0 (NOT side has no voters)."""
    pop = _make_pop(n=NICHE_MIN_SAMPLES + 5)
    # Make every alive slot an AND-expert.
    for s in pop.living_indices().tolist():
        _force_history(pop, int(s), mode=MODE_AND, n_correct=18, n_total=20)
    info = pop.step_window()
    counts = info["species_counts"]
    assert counts["and_expert"] >= NICHE_MIN_SAMPLES
    assert counts["not_expert"] == 0
    assert info["acc_and_swarm"] > 0.5
    assert info["acc_not_swarm"] == 0.0
    assert info["colony_dual_acc"] == 0.0  # one species ⇒ no dual L2 success


def test_colony_dual_acc_nonzero_when_both_species_present() -> None:
    n = 2 * (NICHE_MIN_SAMPLES + 2)
    pop = _make_pop(n=n)
    alive = pop.living_indices().tolist()
    half = len(alive) // 2
    for s in alive[:half]:
        _force_history(pop, int(s), mode=MODE_AND, n_correct=18, n_total=20)
    for s in alive[half:]:
        _force_history(pop, int(s), mode=MODE_NOT, n_correct=18, n_total=20)
    info = pop.step_window()
    counts = info["species_counts"]
    assert counts["and_expert"] >= NICHE_MIN_SAMPLES
    assert counts["not_expert"] >= NICHE_MIN_SAMPLES
    assert info["colony_dual_acc"] > 0.5


# ── 4. SimConfig / runtime API surface ──────────────────────────────────────


def test_sim_config_assortative_temperature_round_trips() -> None:
    from archaea.runtime import SimConfig
    cfg = SimConfig(task=TASK_L2V2, n_initial=4, pop_max=4)
    n = cfg.normalized()
    assert n.assortative_temperature is None  # default = legacy
    n2 = SimConfig(
        task=TASK_L2V2, n_initial=4, pop_max=4,
        assortative_temperature=0.3,
    ).normalized()
    assert n2.assortative_temperature == 0.3


def test_population_assortative_hgt_with_niche_finite_temperature() -> None:
    """Smoke: a Population with finite T runs without crashing and reports T in info."""
    rng = np.random.default_rng(2024)
    slime = SlimeConfig(
        enabled=True, grid_size=8, hgt_enabled=True,
        hgt_prob=0.05, hgt_blend=0.30, hgt_radius=3, hgt_cost=0.0,
        hgt_donor_ratio=1.0, pheromone_bonus_k=0.0, migrate_enabled=False,
    )
    pop = Population(
        pop_max=20, rng=rng, n_initial=20, task=TASK_L2V2,
        slime=slime, assortative_temperature=0.2,
    )
    info = pop.step_window()
    assert info["assortative_temperature"] == 0.2


# ── 5. SPEC_L2_V3.5b — niche-aware consensus, row swarm, inference routing ─


def test_step_window_emits_consensus_and_row_swarm_keys() -> None:
    """Telemetry must surface consensus_acc_swarm + row_acc_swarm + row_n_swarm."""
    pop = _make_pop(n=4)
    info = pop.step_window()
    for key in (
        "consensus_acc_swarm",
        "consensus_bit_swarm",
        "consensus_voters_swarm",
        "acc_and_11_swarm",
        "acc_not_0_swarm",
        "row_acc_swarm",
        "row_n_swarm",
    ):
        assert key in info, key
    rs = info["row_acc_swarm"]
    rn = info["row_n_swarm"]
    assert rs is not None and rn is not None
    assert set(rs.keys()) == {"and_00", "and_01", "and_10", "and_11", "not_a0", "not_a1"}
    # Fresh population: nobody is an expert yet, so every swarm row sample is 0.
    assert all(v == 0 for v in rn.values())


def test_consensus_swarm_zero_voters_when_only_off_niche_species_exist() -> None:
    """A 100% AND-expert colony asked NOT questions has 0 on-niche voters.

    The legacy consensus_acc still reports something (whatever fraction of
    NOT-silent agents accidentally got it right); the swarm version must
    cleanly say "no specialists" via voters_swarm == 0.
    """
    pop = _make_pop(n=NICHE_MIN_SAMPLES + 5)
    # Force every alive slot to be an AND-expert (and zero NOT history).
    for s in pop.living_indices().tolist():
        _force_history(pop, int(s), mode=MODE_AND, n_correct=18, n_total=20)
    # Drive a few step_windows so oracle eventually rolls a NOT question.
    saw_not_zero_voters = False
    for _ in range(40):
        info = pop.step_window()
        oracle = info.get("oracle")
        if oracle and oracle["mode_name"] == "NOT":
            if int(info["consensus_voters_swarm"]) == 0:
                saw_not_zero_voters = True
                assert info["consensus_bit_swarm"] is None
                assert info["consensus_acc_swarm"] == 0.0
                break
    assert saw_not_zero_voters, "expected at least one NOT window with 0 on-niche voters"


# ── 6. top_k_slots_by_niche ─────────────────────────────────────────────────


def test_top_k_slots_by_niche_returns_only_matching_species() -> None:
    pop = _make_pop(n=6)
    alive = pop.living_indices().tolist()
    # Make 2 AND-experts, 2 NOT-experts, 2 novices.
    _force_history(pop, alive[0], mode=MODE_AND, n_correct=18, n_total=20)
    _force_history(pop, alive[1], mode=MODE_AND, n_correct=19, n_total=20)
    _force_history(pop, alive[2], mode=MODE_NOT, n_correct=18, n_total=20)
    _force_history(pop, alive[3], mode=MODE_NOT, n_correct=19, n_total=20)
    # alive[4], alive[5] left as novices (no history).

    and_pick = pop.top_k_slots_by_niche(5, niche="and_expert").tolist()
    not_pick = pop.top_k_slots_by_niche(5, niche="not_expert").tolist()
    assert set(and_pick) == {alive[0], alive[1]}
    assert set(not_pick) == {alive[2], alive[3]}
    # Highest acc_AND first.
    assert and_pick[0] == alive[1]
    assert not_pick[0] == alive[3]


def test_top_k_slots_by_niche_empty_when_no_specialist() -> None:
    pop = _make_pop(n=4)
    # All novices.
    out = pop.top_k_slots_by_niche(3, niche="and_expert")
    assert out.size == 0


# ── 7. Runtime inference routing ────────────────────────────────────────────


def _start_and_freeze_runtime(n: int = 8):
    """Start a runtime then immediately stop the background thread, so we
    can deterministically poke ``_force_history`` without the sim loop
    overwriting the buffers (or starving agents to death).  ``query()``
    itself doesn't need the loop running.
    """
    from archaea.runtime import SimConfig, SimulationRuntime

    cfg = SimConfig(task=TASK_L2V2, n_initial=n, pop_max=n, target_speed_hz=1.0)
    rt = SimulationRuntime()
    rt.start(cfg)
    rt.stop()  # joins the background thread; pop survives
    return rt


def test_runtime_query_target_colony_routes_by_f_s() -> None:
    from archaea.oracle import S_AND_HZ, S_NOT_HZ

    rt = _start_and_freeze_runtime(n=8)
    pop = rt._pop
    assert pop is not None
    alive = pop.living_indices().tolist()
    half = len(alive) // 2
    for s in alive[:half]:
        _force_history(pop, int(s), mode=MODE_AND, n_correct=18, n_total=20)
    for s in alive[half:]:
        _force_history(pop, int(s), mode=MODE_NOT, n_correct=18, n_total=20)

    # 'colony' + AND f_s → and_expert
    r_and = rt.query(
        f_in_hz=75.0, target="colony", top_k=3,
        duration_ms=50.0, warmup_ms=0.0,
        f_b_hz=75.0, f_s_hz=float(S_AND_HZ),
    )
    assert r_and["target_resolved"] == "and_expert"
    assert r_and["target_degraded"] is None
    assert all(int(a["slot"]) in alive[:half] for a in r_and["agents"])

    # 'colony' + NOT f_s → not_expert
    r_not = rt.query(
        f_in_hz=25.0, target="colony", top_k=3,
        duration_ms=50.0, warmup_ms=0.0,
        f_b_hz=25.0, f_s_hz=float(S_NOT_HZ),
    )
    assert r_not["target_resolved"] == "not_expert"
    assert all(int(a["slot"]) in alive[half:] for a in r_not["agents"])


def test_runtime_query_specialist_falls_back_when_niche_empty() -> None:
    from archaea.oracle import S_AND_HZ

    rt = _start_and_freeze_runtime(n=6)
    pop = rt._pop
    assert pop is not None
    # No NOT specialists at all.
    for s in pop.living_indices().tolist():
        _force_history(pop, int(s), mode=MODE_AND, n_correct=18, n_total=20)
    r = rt.query(
        f_in_hz=25.0, target="not_expert", top_k=3,
        duration_ms=50.0, warmup_ms=0.0,
        f_b_hz=25.0, f_s_hz=float(S_AND_HZ),
    )
    assert r["target_resolved"] == "ensemble"
    assert r["target_degraded"] == "no_not_expert_specialist"


# ── 7. SPEC_L2_V3.5c — hard-row qualification (silent-pretender filter) ─────


def test_silent_pretender_does_not_qualify_as_and_expert() -> None:
    """An agent that scores high on AND mode purely by being silent on the easy
    target=0 rows (and never firing on (1,1)) must NOT be classified as
    AND_EXPERT under v3.5c — even though its general AND-mode acc passes 0.65.
    """
    pop = _make_pop(n=2)
    # General buf: 16/20 → 0.80, well above 0.65 mode threshold.
    # Hard buf: 0/8 → 0.00 on the (1,1) row → must be rejected.
    _force_history(
        pop, 0, mode=MODE_AND,
        n_correct=16, n_total=20,
        hard_n_correct=0, hard_n_total=8,
    )
    assert pop._species_of_slot(0) == SPECIES_NOVICE
    # Sanity: the general-mode bar would have passed it.
    assert pop._logic_acc_slot(0, MODE_AND) >= NICHE_SPECIALIST_THRESHOLD
    assert pop._logic_hard_acc_slot(0, MODE_AND) < NICHE_HARD_THRESHOLD


def test_real_and_expert_qualifies_when_hard_row_is_competent() -> None:
    """An agent with both general acc ≥ 0.65 AND hard-row acc ≥ 0.50 on
    (1,1)-AND should still pass as AND_EXPERT.  The new bar must not break the
    legitimate case."""
    pop = _make_pop(n=2)
    _force_history(
        pop, 0, mode=MODE_AND,
        n_correct=18, n_total=20,
        hard_n_correct=8, hard_n_total=10,
    )
    assert pop._species_of_slot(0) == SPECIES_AND_EXPERT


def test_silent_pretender_not_expert_also_rejected() -> None:
    """Symmetric check for NOT side: silent on (a=0) means no NOT_EXPERT badge."""
    pop = _make_pop(n=2)
    _force_history(
        pop, 0, mode=MODE_NOT,
        n_correct=16, n_total=20,
        hard_n_correct=0, hard_n_total=8,
    )
    assert pop._species_of_slot(0) == SPECIES_NOVICE


def test_hard_row_min_samples_gates_label() -> None:
    """Even a perfect hard-row record below NICHE_HARD_MIN_SAMPLES is too thin
    to count as expertise — caller must wait for more (1,1) windows."""
    pop = _make_pop(n=2)
    n_hard = max(0, NICHE_HARD_MIN_SAMPLES - 1)
    _force_history(
        pop, 0, mode=MODE_AND,
        n_correct=18, n_total=20,
        hard_n_correct=n_hard, hard_n_total=n_hard,
    )
    assert pop._species_of_slot(0) == SPECIES_NOVICE


def test_record_logic_window_fills_hard_buffer_on_target_one() -> None:
    """Driving _record_logic_window with target_bit=1 should accumulate the
    hard-row buffer; with target_bit=0 it should leave the hard buffer alone."""
    pop = _make_pop(n=2)
    slot = 0
    pop._record_logic_window(slot, MODE_AND, True, target_bit=1)
    pop._record_logic_window(slot, MODE_AND, False, target_bit=1)
    pop._record_logic_window(slot, MODE_AND, True, target_bit=0)  # easy row, must NOT touch hard buf
    assert pop._acc_and_hard_n[slot] == 2
    assert pop._acc_and_n[slot] == 3
    # 1 correct out of 2 hard windows → 0.5
    assert pop._logic_hard_acc_slot(slot, MODE_AND) == 0.5


def test_top_k_slots_by_niche_ranks_hard_competent_first() -> None:
    """Among AND specialists, the agent with higher (1,1) competence should
    rank above one with the same general acc but worse hard-row score."""
    pop = _make_pop(n=4)
    alive = pop.living_indices().tolist()[:2]
    a, b = int(alive[0]), int(alive[1])
    # Same general acc 18/20, but a is much better on the hard row than b.
    _force_history(pop, a, mode=MODE_AND, n_correct=18, n_total=20,
                   hard_n_correct=10, hard_n_total=10)
    _force_history(pop, b, mode=MODE_AND, n_correct=18, n_total=20,
                   hard_n_correct=5, hard_n_total=10)
    picks = pop.top_k_slots_by_niche(2, "and_expert")
    assert picks.tolist()[0] == a, f"expected hard-competent slot {a} first, got {picks.tolist()}"

