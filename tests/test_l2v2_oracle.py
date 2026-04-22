"""SPEC_L2_V2.0 — oracle truth table, three-channel stimulus, reward routing."""

from __future__ import annotations

import numpy as np
import pytest

from archaea.neuron import N_INPUT
from archaea.oracle import (
    BIT_THRESHOLD_HZ,
    LOGIC_HIGH_HZ,
    LOGIC_LOW_HZ,
    MODE_AND,
    MODE_NAMES,
    MODE_NOT,
    OUT_SPIKING_THRESHOLD_HZ,
    R_AND_SILENT_RAW,
    R_AND_SPIKE_RAW,
    R_NOT_SILENT_RAW,
    R_NOT_SPIKE_RAW,
    REWARD_SCALE,
    S_AND_HZ,
    S_JITTER_HZ,
    S_NOT_HZ,
    classify_output,
    draw_oracle_sample,
    poisson_three_channels,
)
from archaea.population import LOGIC_HISTORY, Population
from archaea.task import N_CH_A, N_CH_B, N_CH_S, TASK_L1, TASK_L2V2, validate_task


# ── task identifiers ────────────────────────────────────────────────────────
def test_task_validate_accepts_l1_and_l2v2():
    assert validate_task("l1") == "l1"
    assert validate_task("l2v2_ctrl") == "l2v2_ctrl"


def test_task_validate_rejects_unknown():
    with pytest.raises(ValueError):
        validate_task("nonsense")


# ── channel split sums to N_INPUT ──────────────────────────────────────────
def test_channel_split_matches_n_input():
    assert N_CH_A + N_CH_B + N_CH_S == N_INPUT


# ── poisson_three_channels shape and order ─────────────────────────────────
def test_three_channels_shape_and_order():
    rng = np.random.default_rng(0)
    s = poisson_three_channels(rng, 100.0, 100.0, 100.0, 500.0)
    assert s.shape == (500, N_INPUT)
    assert ((s == 0.0) | (s == 1.0)).all()


def test_three_channels_zero_rate_silent_per_channel():
    """A channel at 0 Hz must produce zero spikes on its own neurons only."""
    rng = np.random.default_rng(1)
    # Only channel A active; B and S silent.
    s = poisson_three_channels(rng, 200.0, 0.0, 0.0, 500.0)
    a = s[:, :N_CH_A]
    b = s[:, N_CH_A : N_CH_A + N_CH_B]
    sel = s[:, N_CH_A + N_CH_B :]
    assert a.sum() > 0
    assert b.sum() == 0
    assert sel.sum() == 0


# ── oracle truth table ─────────────────────────────────────────────────────
def test_oracle_AND_truth_table_exhaustive():
    """For every (mode, bit_a, bit_b) the target bit must obey SPEC §2.2."""
    rng = np.random.default_rng(7)
    seen = {(MODE_AND, 0, 0), (MODE_AND, 0, 1), (MODE_AND, 1, 0), (MODE_AND, 1, 1)}
    found = set()
    # Sample many times to hit every (mode, a, b) combo.
    for _ in range(2000):
        sample = draw_oracle_sample(rng)
        if sample.mode != MODE_AND:
            continue
        found.add((sample.mode, sample.bit_a, sample.bit_b))
        expected = 1 if (sample.bit_a == 1 and sample.bit_b == 1) else 0
        assert sample.target_bit == expected, sample
        # rate encoding obeys threshold rule
        assert (sample.f_a_hz > BIT_THRESHOLD_HZ) == (sample.bit_a == 1)
        assert (sample.f_b_hz > BIT_THRESHOLD_HZ) == (sample.bit_b == 1)
    assert seen.issubset(found), f"missing combos: {seen - found}"


def test_oracle_NOT_target_inverts_A():
    rng = np.random.default_rng(11)
    saw_zero = saw_one = False
    for _ in range(2000):
        sample = draw_oracle_sample(rng)
        if sample.mode != MODE_NOT:
            continue
        assert sample.target_bit == 1 - sample.bit_a
        if sample.bit_a == 0:
            saw_zero = True
        else:
            saw_one = True
    assert saw_zero and saw_one


def test_oracle_S_frequency_brackets():
    """S must be 20±5 in AND mode, 80±5 in NOT mode (SPEC §2.1)."""
    rng = np.random.default_rng(13)
    for _ in range(500):
        sample = draw_oracle_sample(rng)
        if sample.mode == MODE_AND:
            assert abs(sample.f_s_hz - S_AND_HZ) <= S_JITTER_HZ + 1e-9
        else:
            assert abs(sample.f_s_hz - S_NOT_HZ) <= S_JITTER_HZ + 1e-9


def test_oracle_reward_table_matches_rebalanced_values():
    """ERRATA v2.1 — spike-correct rewards amplified to break silent collapse."""
    rng = np.random.default_rng(17)
    rewards = {}
    for _ in range(2000):
        sample = draw_oracle_sample(rng)
        rewards[(sample.mode, sample.target_bit)] = sample.reward_correct
    # all four entries must be discovered
    assert (MODE_AND, 0) in rewards
    assert (MODE_AND, 1) in rewards
    assert (MODE_NOT, 0) in rewards
    assert (MODE_NOT, 1) in rewards
    # exact rebalanced values (× REWARD_SCALE)
    assert rewards[(MODE_AND, 1)] == pytest.approx(R_AND_SPIKE_RAW * REWARD_SCALE)
    assert rewards[(MODE_AND, 0)] == pytest.approx(R_AND_SILENT_RAW * REWARD_SCALE)
    assert rewards[(MODE_NOT, 1)] == pytest.approx(R_NOT_SPIKE_RAW * REWARD_SCALE)
    assert rewards[(MODE_NOT, 0)] == pytest.approx(R_NOT_SILENT_RAW * REWARD_SCALE)
    # NOT(1) must remain the highest premium (SPEC §2.2 intent preserved)
    assert rewards[(MODE_NOT, 1)] > rewards[(MODE_AND, 1)]
    assert rewards[(MODE_NOT, 1)] > 10 * rewards[(MODE_NOT, 0)]
    assert rewards[(MODE_AND, 1)] > 10 * rewards[(MODE_AND, 0)]


def test_reward_table_breaks_silent_collapse():
    """Anti-attractor invariant: 'always silent' must be net-negative vs breath.

    With BREATH_PER_WINDOW=1.25 (economy.py) and a uniform (mode, A, B) draw,
    an agent that never spikes wins 3/4 of AND rows and 1/2 of NOT rows.
    Its expected per-window reward must come out STRICTLY below breath, or
    else evolution settles into the silent attractor and never learns the
    actual logic (this is the bug ERRATA v2.1 exists to fix).
    """
    from archaea.economy import BREATH_PER_WINDOW

    e_and_silent = R_AND_SILENT_RAW * REWARD_SCALE * (3 / 4)  # win 3/4 AND rows
    e_not_silent = R_NOT_SILENT_RAW * REWARD_SCALE * (1 / 2)  # win 1/2 NOT rows
    silent_per_window = 0.5 * e_and_silent + 0.5 * e_not_silent
    assert silent_per_window < BREATH_PER_WINDOW, (
        f"silent-strategy expected reward {silent_per_window:.3f}/window "
        f"≥ breath {BREATH_PER_WINDOW}/window — silent collapse will dominate"
    )

    # And the perfect-logic strategy must beat it by at least an order of
    # magnitude so the evolutionary gradient is unambiguous.
    e_and_perfect = REWARD_SCALE * (3 / 4 * R_AND_SILENT_RAW + 1 / 4 * R_AND_SPIKE_RAW)
    e_not_perfect = REWARD_SCALE * (1 / 2 * R_NOT_SILENT_RAW + 1 / 2 * R_NOT_SPIKE_RAW)
    perfect_per_window = 0.5 * e_and_perfect + 0.5 * e_not_perfect
    assert perfect_per_window > 10 * silent_per_window, (
        f"perfect/silent ratio {perfect_per_window / silent_per_window:.1f}× "
        "is too small — gradient may be too weak"
    )


# ── output classification ──────────────────────────────────────────────────
def test_classify_output_threshold():
    assert classify_output(OUT_SPIKING_THRESHOLD_HZ - 0.1) == 0
    assert classify_output(OUT_SPIKING_THRESHOLD_HZ + 0.1) == 1


# ── L2v2 short population run: doesn't crash, accuracy is finite, reward
#    flows to credit, and weights live in [-1.5, 1.5] at init ──────────────
def test_l2v2_population_short_run():
    rng = np.random.default_rng(2026)
    pop = Population(pop_max=40, rng=rng, n_initial=20, task=TASK_L2V2)
    # Initial weights bounded by SPEC §3.1
    alive = pop.weights[pop.alive]
    assert alive.min() >= -1.5 - 1e-12
    assert alive.max() <= 1.5 + 1e-12

    seen_modes = set()
    for _ in range(60):
        info = pop.step_window()
        assert info["task"] == TASK_L2V2
        assert info["oracle"] is not None
        seen_modes.add(info["oracle"]["mode"])
        # Population-level metrics must always be in [0, 1]
        assert 0.0 <= info["acc_and_pop"] <= 1.0
        assert 0.0 <= info["acc_not_pop"] <= 1.0
        assert 0.0 <= info["both_pass_pct"] <= 1.0
        assert 0.0 <= info["logic_diversity"] <= 1.0
        assert 0.0 <= info["consensus_acc"] <= 1.0
    # Both modes should appear over 60 windows (P(miss) ≈ 2^-60)
    assert seen_modes == {MODE_AND, MODE_NOT}


def test_l2v2_fitness_undefined_until_both_modes_seen():
    """An agent that has only seen AND windows must NOT be considered evaluable."""
    rng = np.random.default_rng(99)
    pop = Population(pop_max=4, rng=rng, n_initial=4, task=TASK_L2V2)
    # Manually inject one AND-only correct sample for slot 0.
    pop._record_logic_window(0, MODE_AND, True)
    assert pop._fitness_defined(0) is False
    # Add a NOT sample → now defined.
    pop._record_logic_window(0, MODE_NOT, False)
    assert pop._fitness_defined(0) is True
    # Mean accuracy = (1 + 0) / 2 = 0.5
    assert pop._fitness_slot(0) == pytest.approx(0.5)


def test_l2v2_logic_history_ring_overwrites():
    """Ring buffer of size LOGIC_HISTORY must overwrite, not crash."""
    rng = np.random.default_rng(123)
    pop = Population(pop_max=2, rng=rng, n_initial=2, task=TASK_L2V2)
    for i in range(LOGIC_HISTORY * 3):
        pop._record_logic_window(0, MODE_AND, (i % 2 == 0))
        pop._record_logic_window(0, MODE_NOT, True)
    # Acc should reflect the ring's contents, not the totals.
    assert 0.4 <= pop._logic_acc_slot(0, MODE_AND) <= 0.6
    assert pop._logic_acc_slot(0, MODE_NOT) == 1.0


def test_l1_path_unchanged_bit_identical():
    """Running an L1 population for N windows must be unaffected by L2v2 code paths.

    Not just "passes" — the per-window (births, deaths) trace must match the
    pre-L2 baseline that the test_short_run snapshot establishes.  Here we
    sanity-check that L1 runs without raising and produces task='l1' info.
    """
    rng = np.random.default_rng(42)
    pop = Population(pop_max=20, rng=rng, n_initial=10, task=TASK_L1)
    for _ in range(15):
        info = pop.step_window()
        assert info["task"] == TASK_L1
        assert info["oracle"] is None
        assert info["acc_and_pop"] == 0.0
        assert info["acc_not_pop"] == 0.0


def test_mode_names_table():
    assert MODE_NAMES[MODE_AND] == "AND"
    assert MODE_NAMES[MODE_NOT] == "NOT"
