"""SPEC_L2_V2.0 — oracle truth table, three-channel stimulus, reward routing."""

from __future__ import annotations

import numpy as np
import pytest

from archaea.neuron import N_INPUT
from archaea.oracle import (
    BIT_THRESHOLD_HZ,
    DEFAULT_TASK_DIFFICULTY,
    LOGIC_HIGH_HZ,
    LOGIC_LOW_HZ,
    MODE_AND,
    MODE_NAMES,
    MODE_NOT,
    OUT_SPIKING_THRESHOLD_HZ,
    P_AND_TARGET_ONE,
    P_MODE_AND,
    P_NOT_TARGET_ONE,
    R_AND_SILENT_RAW,
    R_AND_SPIKE_RAW,
    R_NOT_SILENT_RAW,
    R_NOT_SPIKE_RAW,
    REWARD_SCALE,
    S_AND_HZ,
    S_JITTER_HZ,
    S_NOT_HZ,
    SPIKE_EFFORT_BONUS_K,
    TASK_DIFFICULTY_PRESETS,
    classify_output,
    difficulty_weights,
    draw_oracle_sample,
    poisson_three_channels,
    spike_effort_bonus,
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
    """ERRATA v2.2 — spike-correct rewards amplified, silent kept viable."""
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
    # spike rewards dominate silent rewards by a wide margin
    assert rewards[(MODE_AND, 1)] > 5 * rewards[(MODE_AND, 0)]
    assert rewards[(MODE_NOT, 1)] > 5 * rewards[(MODE_NOT, 0)]


def test_weighted_sampling_target_one_is_50pct():
    """ERRATA v2.3 — environment must produce target=1 in ~50% of windows.

    This is the load-bearing invariant that breaks the silent attractor:
    if target=1 occurs only 37.5% (the v2.2 uniform default), permanent
    silence still hits acc_AND≈75% and looks like learning.  At 50%
    target=1, silent acc_AND falls to 50% — indistinguishable on the
    dashboard from random guessing, exposing the lazy strategy.
    """
    rng = np.random.default_rng(2025)
    N = 8000
    n_target_one = 0
    n_and = 0
    n_and_target_one = 0
    n_not = 0
    n_not_target_one = 0
    for _ in range(N):
        s = draw_oracle_sample(rng)
        if s.target_bit == 1:
            n_target_one += 1
        if s.mode == MODE_AND:
            n_and += 1
            if s.target_bit == 1:
                n_and_target_one += 1
        else:
            n_not += 1
            if s.target_bit == 1:
                n_not_target_one += 1

    # Marginal P(target=1) — the headline invariant
    p_one = n_target_one / N
    assert 0.46 <= p_one <= 0.54, (
        f"P(target=1)={p_one:.3f}, expected ≈0.50 (v2.3) — silent ceiling not broken"
    )

    # Mode marginals match config
    p_and = n_and / N
    assert abs(p_and - P_MODE_AND) < 0.04, f"P(AND)={p_and:.3f} vs {P_MODE_AND}"

    # Conditional P(target=1 | AND) and P(target=1 | NOT)
    p_and_one = n_and_target_one / max(n_and, 1)
    p_not_one = n_not_target_one / max(n_not, 1)
    assert abs(p_and_one - P_AND_TARGET_ONE) < 0.04, (
        f"P(target=1|AND)={p_and_one:.3f} vs {P_AND_TARGET_ONE}"
    )
    assert abs(p_not_one - P_NOT_TARGET_ONE) < 0.04, (
        f"P(target=1|NOT)={p_not_one:.3f} vs {P_NOT_TARGET_ONE}"
    )


def test_weighted_sampling_three_target_zero_AND_rows_balanced():
    """The 3 target=0 AND rows ((0,0),(0,1),(1,0)) must split the 50% mass evenly.

    If one of them is starved, agents could win on the others without
    learning the full AND truth table — defeating the purpose.
    """
    rng = np.random.default_rng(99)
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    n_and = 0
    for _ in range(8000):
        s = draw_oracle_sample(rng)
        if s.mode != MODE_AND:
            continue
        n_and += 1
        counts[(s.bit_a, s.bit_b)] += 1
    # (1,1) should be ~50% of AND mass; the other 3 each ~16.7%
    assert counts[(1, 1)] / n_and > 0.45
    for row in [(0, 0), (0, 1), (1, 0)]:
        share = counts[row] / n_and
        assert 0.12 <= share <= 0.22, (
            f"AND row {row} share {share:.3f} too far from 1/6 — imbalanced"
        )


def test_v23_silent_starves_perfect_dominates():
    """v2.3 + v2.4 ecological invariants under WEIGHTED sampling.

    1. silent_per_window < BREATH_PER_WINDOW
       ↑ With P(target=1)=0.5, silent agents only collect base reward on
       the 50% of windows that expect 0; v2.4's effort-bonus (capped at K
       per window) doesn't rescue them.  They still net negative, slowly
       starving over ~3 min and freeing生态位 for true-logic descendants.
       This is intentional: the founder-collapse risk that doomed v2.1
       no longer applies because we only switch to v2.3 *after* an elite
       sub-population has been established.

    2. perfect_per_window ≥ 8 × silent_per_window
       ↑ True-logic reproduction rate must dwarf silent's so they fill
       the void left by starving silent agents within seconds.  The v2.4
       effort bonus is small (K=0.1) by design and cannot close the gap.
    """
    from archaea.economy import BREATH_PER_WINDOW

    # silent: f_out = 0.  Base reward only on target=0 windows.
    # Effort bonus: target=0 → K; target=1 → 0.  Marginal P(target=1)=0.5
    # ⇒ expected effort bonus = 0.5 · K.
    p_and_silent_correct = (1 - P_AND_TARGET_ONE)        # silent right on AND target=0 rows
    p_not_silent_correct = (1 - P_NOT_TARGET_ONE)        # silent right on NOT target=0 rows
    e_and_silent = R_AND_SILENT_RAW * REWARD_SCALE * p_and_silent_correct
    e_not_silent = R_NOT_SILENT_RAW * REWARD_SCALE * p_not_silent_correct
    silent_base = P_MODE_AND * e_and_silent + (1 - P_MODE_AND) * e_not_silent
    p_target_one = (
        P_MODE_AND * P_AND_TARGET_ONE + (1 - P_MODE_AND) * P_NOT_TARGET_ONE
    )
    silent_effort = (1.0 - p_target_one) * SPIKE_EFFORT_BONUS_K
    silent_per_window = silent_base + silent_effort

    # Perfect agent: f_out > threshold on target=1, f_out = 0 on target=0.
    # Effort bonus saturates at K on every window (correct direction either way).
    e_and_perfect = REWARD_SCALE * (
        (1 - P_AND_TARGET_ONE) * R_AND_SILENT_RAW
        + P_AND_TARGET_ONE * R_AND_SPIKE_RAW
    )
    e_not_perfect = REWARD_SCALE * (
        (1 - P_NOT_TARGET_ONE) * R_NOT_SILENT_RAW
        + P_NOT_TARGET_ONE * R_NOT_SPIKE_RAW
    )
    perfect_base = P_MODE_AND * e_and_perfect + (1 - P_MODE_AND) * e_not_perfect
    perfect_per_window = perfect_base + SPIKE_EFFORT_BONUS_K

    # Soft-starve invariant — silent still loses credit even with effort bonus
    assert silent_per_window < BREATH_PER_WINDOW, (
        f"silent expected reward {silent_per_window:.3f}/window "
        f"≥ breath {BREATH_PER_WINDOW}/window — v2.4 effort bonus is too generous"
    )
    # Not too punishing: silent should survive ≥150 windows from starting credit (50)
    starvation_rate = BREATH_PER_WINDOW - silent_per_window
    assert starvation_rate < 0.5, (
        f"silent loses {starvation_rate:.3f}/win — too fast, risks founder collapse"
    )

    # Dominance invariant — perfect must reproduce ~10× faster than silent
    assert perfect_per_window > 8 * silent_per_window, (
        f"perfect/silent ratio {perfect_per_window / silent_per_window:.1f}× "
        "is too small — silent could persist as a subpopulation"
    )


# ── v2.3.1 difficulty presets (environment-shaping slider) ─────────────────
def test_difficulty_presets_registry_complete():
    """All four named presets must resolve and carry the four required keys
    (v3.2 added reward_wrong; sampling probabilities still in [0,1])."""
    expected_keys = {
        "p_mode_and", "p_and_target_one", "p_not_target_one", "reward_wrong",
    }
    prob_keys = {"p_mode_and", "p_and_target_one", "p_not_target_one"}
    for name in ("uniform", "balanced", "hard", "extreme"):
        w = difficulty_weights(name)
        assert set(w.keys()) == expected_keys
        for k in prob_keys:
            assert 0.0 <= w[k] <= 1.0, f"{name}.{k}={w[k]} out of [0,1]"
    assert DEFAULT_TASK_DIFFICULTY in TASK_DIFFICULTY_PRESETS


def test_difficulty_presets_monotone_target_one():
    """Higher-difficulty presets must increase P(target=1) on AND mode.
    This is the contract the SetupPage slider hint relies on.
    """
    p_uniform = TASK_DIFFICULTY_PRESETS["uniform"]["p_and_target_one"]
    p_balanced = TASK_DIFFICULTY_PRESETS["balanced"]["p_and_target_one"]
    p_hard = TASK_DIFFICULTY_PRESETS["hard"]["p_and_target_one"]
    p_extreme = TASK_DIFFICULTY_PRESETS["extreme"]["p_and_target_one"]
    assert p_uniform < p_balanced < p_hard < p_extreme


def test_difficulty_unknown_raises():
    with pytest.raises(ValueError):
        difficulty_weights("godmode")


def test_draw_oracle_sample_honors_weights_uniform_mode():
    """In 'uniform' preset, P(target=1|AND) ≈ 0.25 (SPEC original) — the silent
    ceiling is high (75%), which is exactly the "用户能选回旧 SPEC" 的意图.
    """
    rng = np.random.default_rng(7)
    w = difficulty_weights("uniform")
    n_and = 0
    n_and_one = 0
    for _ in range(8000):
        s = draw_oracle_sample(rng, **w)
        if s.mode == MODE_AND:
            n_and += 1
            if s.target_bit == 1:
                n_and_one += 1
    p = n_and_one / max(n_and, 1)
    assert 0.21 <= p <= 0.30, f"P(target=1|AND, uniform)={p:.3f} ≠ ~0.25"


def test_draw_oracle_sample_honors_weights_extreme_mode():
    """'extreme' must drive P(target=1) close to 0.9 on both modes."""
    rng = np.random.default_rng(8)
    w = difficulty_weights("extreme")
    n_and = n_and_one = n_not = n_not_one = 0
    for _ in range(8000):
        s = draw_oracle_sample(rng, **w)
        if s.mode == MODE_AND:
            n_and += 1
            if s.target_bit == 1:
                n_and_one += 1
        else:
            n_not += 1
            if s.target_bit == 1:
                n_not_one += 1
    assert (n_and_one / max(n_and, 1)) > 0.85
    assert (n_not_one / max(n_not, 1)) > 0.85


# ── output classification ──────────────────────────────────────────────────
def test_classify_output_threshold():
    assert classify_output(OUT_SPIKING_THRESHOLD_HZ - 0.1) == 0
    assert classify_output(OUT_SPIKING_THRESHOLD_HZ + 0.1) == 1


# ── v2.4 platform-cliff fix: spike threshold lowered + effort bonus ────────
def test_v24_spike_threshold_lowered_to_20hz():
    """v2.4 (A): the spike threshold must be ≤25 Hz so the near-spike cohort
    observed in v2.3 diagnostics (f_out ≈ 26-32 Hz on (1,1)) actually
    crosses into "judged as 1" territory.  Setting it back to 50 would
    re-introduce the platform-cliff fitness landscape.
    """
    assert OUT_SPIKING_THRESHOLD_HZ <= 25.0, (
        f"OUT_SPIKING_THRESHOLD_HZ={OUT_SPIKING_THRESHOLD_HZ} — v2.4 platform-cliff fix reverted"
    )
    # Sanity bound: also strictly above LOGIC_LOW_HZ (25 Hz) /4 so spurious
    # noise floor doesn't cross it on its own.
    assert OUT_SPIKING_THRESHOLD_HZ > 5.0


def test_spike_effort_bonus_target_one_monotone_increasing():
    """target=1: bonus must rise monotonically from 0 (silent) to K (≥threshold)."""
    K = SPIKE_EFFORT_BONUS_K
    assert spike_effort_bonus(0.0, target_bit=1) == pytest.approx(0.0)
    half = spike_effort_bonus(OUT_SPIKING_THRESHOLD_HZ * 0.5, target_bit=1)
    full = spike_effort_bonus(OUT_SPIKING_THRESHOLD_HZ, target_bit=1)
    over = spike_effort_bonus(OUT_SPIKING_THRESHOLD_HZ * 5.0, target_bit=1)
    assert half == pytest.approx(0.5 * K)
    assert full == pytest.approx(K)
    assert over == pytest.approx(K), "must saturate at threshold (no double-pay with table reward)"


def test_spike_effort_bonus_target_zero_monotone_decreasing():
    """target=0: bonus must fall monotonically from K (silent) to 0 (≥threshold)."""
    K = SPIKE_EFFORT_BONUS_K
    assert spike_effort_bonus(0.0, target_bit=0) == pytest.approx(K)
    half = spike_effort_bonus(OUT_SPIKING_THRESHOLD_HZ * 0.5, target_bit=0)
    full = spike_effort_bonus(OUT_SPIKING_THRESHOLD_HZ, target_bit=0)
    over = spike_effort_bonus(OUT_SPIKING_THRESHOLD_HZ * 5.0, target_bit=0)
    assert half == pytest.approx(0.5 * K)
    assert full == pytest.approx(0.0)
    assert over == pytest.approx(0.0)


def test_spike_effort_bonus_negative_clamped():
    """Defensive: negative f_out (shouldn't happen) clamps to 0."""
    K = SPIKE_EFFORT_BONUS_K
    assert spike_effort_bonus(-100.0, target_bit=1) == pytest.approx(0.0)
    assert spike_effort_bonus(-100.0, target_bit=0) == pytest.approx(K)


def test_spike_effort_bonus_directional_gradient_breaks_platform_cliff():
    """The headline guarantee of v2.4 (C): an agent attempting a spike on a
    target=1 window must earn STRICTLY MORE per-window than a fully silent
    agent on the same window — even when both are *misclassified* as '0'.

    This is the upward gradient that v2.3 was missing.  Without it, the
    population had no incentive to keep mutations that produced f_out=10
    Hz (judged as 0, same outcome as f_out=0).
    """
    # Imagine two identical-fate agents on a target=1 window, both judged 0:
    silent_bonus = spike_effort_bonus(0.0, target_bit=1)
    near_spike_bonus = spike_effort_bonus(OUT_SPIKING_THRESHOLD_HZ * 0.8, target_bit=1)
    assert near_spike_bonus > silent_bonus, (
        "no upward gradient for sub-threshold spike attempts — platform-cliff is back"
    )


def test_l2v2_population_step_includes_effort_bonus_in_reward():
    """The effort bonus must actually flow into per-window credit deltas, not
    just exist as an unused function.  We zero out a population's weights so
    every agent is silent (f_out=0) and confirm the per-window reward on a
    target=0 window equals base reward + K (full silent bonus).
    """
    rng = np.random.default_rng(31415)
    pop = Population(
        pop_max=8, rng=rng, n_initial=8,
        task=TASK_L2V2, task_difficulty="balanced",
    )
    # Force every agent silent.
    pop.weights[:] = 0.0
    # Run several windows and verify on each that the reward telemetry
    # matches base reward + spike_effort_bonus(0, target_bit) for f_out=0.
    saw_target_zero = saw_target_one = False
    for _ in range(40):
        info = pop.step_window()
        oracle = info["oracle"]
        assert oracle is not None
        target = int(oracle["target_bit"])
        # All agents are silent → output bit = 0 → "correct" iff target == 0.
        if target == 0:
            saw_target_zero = True
            base = float(oracle["reward_correct"])
            expected = base + SPIKE_EFFORT_BONUS_K          # full silent bonus
        else:
            saw_target_one = True
            base = 0.0                                       # reward_wrong
            expected = base + 0.0                            # silent on target=1 ⇒ no bonus
        # Every alive agent must show the same per-window reward (deterministic
        # for f_out = 0; agents are identical with weights=0).
        alive_mask = pop.alive
        observed = pop._last_reward[alive_mask]
        assert np.allclose(observed, expected, atol=1e-9), (
            f"target={target} expected reward {expected:.3f} got {observed[:3]}"
        )
        if saw_target_zero and saw_target_one:
            break
    assert saw_target_zero and saw_target_one, "needed both target bits to validate bonus paths"


# ── v2.5 prebiotic-stage founder bias (companion fix to v2.4) ──────────────
def test_v25_founder_weights_in_offset_range():
    """v2.5: every initial weight must lie in [-0.5, 1.5] (the prebiotic-
    stage offset range) for L2v2.  This is the small-sample boundary check;
    a separate test pins the SPEC §3.1 negative-share invariant.
    """
    from archaea.population import L2V2_WEIGHT_INIT_HIGH, L2V2_WEIGHT_INIT_LOW
    assert L2V2_WEIGHT_INIT_LOW == -0.5, (
        f"v2.5 founder offset reverted: LOW={L2V2_WEIGHT_INIT_LOW}"
    )
    assert L2V2_WEIGHT_INIT_HIGH == 1.5
    rng = np.random.default_rng(2025)
    pop = Population(pop_max=200, rng=rng, n_initial=200, task=TASK_L2V2)
    w = pop.weights[pop.alive]
    assert w.min() >= L2V2_WEIGHT_INIT_LOW - 1e-12
    assert w.max() <= L2V2_WEIGHT_INIT_HIGH + 1e-12


def test_v25_founder_negative_weight_share_meets_spec():
    """SPEC §3.1: ≥20% of weights must be negative for inhibitory paths.

    Uniform(-0.5, 1.5) has theoretical negative share = 0.5/2.0 = 25%.
    With N=200 founders × 220 weights = 44000 samples the empirical share
    will be tightly centered on 0.25; we accept anything ≥0.20 (the SPEC
    floor — a future tweak that drops below this floor must be rejected).
    """
    rng = np.random.default_rng(2025)
    pop = Population(pop_max=200, rng=rng, n_initial=200, task=TASK_L2V2)
    w = pop.weights[pop.alive].ravel()
    neg_share = float((w < 0.0).sum()) / w.size
    assert neg_share >= 0.20, (
        f"negative weight share {neg_share:.3f} < SPEC §3.1 floor 0.20 — "
        "inhibitory paths can't be discovered by mutation"
    )
    # Sanity: shouldn't have *too* many negatives either, or the founder bias
    # is gone and we're back to the v2.0–v2.4 evolvability=0 deadlock.
    assert neg_share <= 0.35, (
        f"negative weight share {neg_share:.3f} too close to symmetric — "
        "v2.5 prebiotic offset has been weakened past usefulness"
    )


def test_v25_founder_mean_weight_positive():
    """v2.5 design intent: E[Σw] should be positive, not 0, so the average
    founder has a non-zero hidden-layer drive.  Theoretical mean of
    Uniform(-0.5, 1.5) = 0.5; empirical should be very close.
    """
    rng = np.random.default_rng(2025)
    pop = Population(pop_max=200, rng=rng, n_initial=200, task=TASK_L2V2)
    w = pop.weights[pop.alive]
    mean_w = float(w.mean())
    # Theoretical: 0.5; with 44000 samples std of mean ≈ sqrt(1/3 / 44000) ≈ 0.003.
    assert 0.45 <= mean_w <= 0.55, (
        f"founder weight mean {mean_w:.3f} not centred on 0.5 — "
        "v2.5 prebiotic offset broken"
    )


def test_v25_founder_population_breaks_silent_deadlock():
    """The headline v2.5 guarantee, end-to-end:

    A fresh L2v2 colony seeded with v2.5 weights MUST produce at least
    *some* spike-correct activity within the first ~10 windows — i.e. some
    individual must cross OUT_SPIKING_THRESHOLD_HZ on at least one (1,1)
    AND or NOT(0) target=1 stimulus.

    This is the test that would have caught the v2.4-only crash where
    the colony died at t_sim=311 s with row_acc.and_11 = 0 (founders
    sat at f_out=0 forever).  If this test fails, the silent deadlock is
    back and the evolutionary loop won't start.
    """
    rng = np.random.default_rng(2026)
    # 200 founders, synapse_gain=2.0 (the realistic deployment setting).
    pop = Population(
        pop_max=400, rng=rng, n_initial=200,
        task=TASK_L2V2, task_difficulty="balanced",
        synapse_gain=2.0,
    )
    # Run 30 windows — long enough to hit several target=1 stimuli of each type.
    saw_correct_target_one = False
    for _ in range(30):
        info = pop.step_window()
        oracle = info["oracle"]
        assert oracle is not None
        # On any target=1 window, count how many agents got it right.
        if oracle["target_bit"] == 1 and info["consensus_acc"] > 0.0:
            # consensus_acc > 0 ⇒ at least one agent answered 1 correctly,
            # which means ≥1 agent emitted f_out > OUT_SPIKING_THRESHOLD_HZ
            # while the oracle was demanding 1 — exactly the spike capability
            # the v2.4 founders lacked.
            saw_correct_target_one = True
            break
    assert saw_correct_target_one, (
        "no founder spiked correctly on a target=1 stimulus in 30 windows — "
        "v2.5 prebiotic offset isn't producing day-zero spikers; the silent "
        "deadlock that killed the v2.4 colony at t_sim=311 s is back"
    )


# ── L2v2 short population run: doesn't crash, accuracy is finite, reward
#    flows to credit, and weights live in [-0.5, 1.5] at init (v2.5 range) ──
def test_l2v2_population_short_run():
    rng = np.random.default_rng(2026)
    pop = Population(pop_max=40, rng=rng, n_initial=20, task=TASK_L2V2)
    # Initial weights bounded by SPEC §3.1 + v2.5 prebiotic offset:
    # Uniform(-0.5, 1.5) — see L2V2_WEIGHT_INIT_LOW/HIGH in population.py.
    alive = pop.weights[pop.alive]
    assert alive.min() >= -0.5 - 1e-12
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
        # v2.3.1 row-specific telemetry
        assert 0.0 <= info["acc_and_11_pop"] <= 1.0
        assert 0.0 <= info["acc_not_0_pop"] <= 1.0
        assert info["row_acc"] is not None
        assert info["row_n"] is not None
        assert info["task_difficulty"] == DEFAULT_TASK_DIFFICULTY
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


# ── ERRATA v3.1 — specialist-mode fitness gating ───────────────────────────
def test_l2v2_specialist_and_only_fitness_defined_after_one_and_sample():
    """In and_only dish, fitness is defined as soon as the agent has one AND
    sample — requiring NOT samples (which never come) used to freeze the
    entire population at sigma=SIGMA_BASE forever."""
    rng = np.random.default_rng(7)
    pop = Population(
        pop_max=4, rng=rng, n_initial=4, task=TASK_L2V2,
        task_difficulty="and_only",
    )
    assert pop._fitness_defined(0) is False
    pop._record_logic_window(0, MODE_AND, True)
    assert pop._fitness_defined(0) is True
    # Specialist fitness reflects only the active mode (full 1.0, not 0.5).
    assert pop._fitness_slot(0) == pytest.approx(1.0)


def test_l2v2_specialist_not_only_fitness_defined_after_one_not_sample():
    rng = np.random.default_rng(8)
    pop = Population(
        pop_max=4, rng=rng, n_initial=4, task=TASK_L2V2,
        task_difficulty="not_only",
    )
    assert pop._fitness_defined(0) is False
    pop._record_logic_window(0, MODE_NOT, True)
    assert pop._fitness_defined(0) is True
    assert pop._fitness_slot(0) == pytest.approx(1.0)


def test_l2v2_specialist_global_sigma_anneals():
    """Once specialist agents have valid fitness, sigma must drop below
    SIGMA_BASE — the headline symptom of the v3.1 bug was sigma locked at
    its max because mean_f stayed 0 for the whole run."""
    from archaea.population import SIGMA_BASE
    rng = np.random.default_rng(9)
    pop = Population(
        pop_max=4, rng=rng, n_initial=4, task=TASK_L2V2,
        task_difficulty="not_only",
    )
    # Inject perfect NOT for all 4 living slots.
    for slot in range(4):
        pop._record_logic_window(slot, MODE_NOT, True)
    sigma = pop.global_sigma()
    # exp(-2 * 1.0) ≈ 0.135 → expect sigma ≈ 0.135 * SIGMA_BASE, well below max.
    assert sigma < 0.5 * SIGMA_BASE


def test_l2v2_mixed_dish_still_requires_both_modes():
    """Regression: the v3.1 carve-out must NOT loosen the contract for
    mixed dishes (the original v2.0 guarantee that a brand-new agent who
    has only seen AND windows is treated as undefined)."""
    rng = np.random.default_rng(10)
    pop = Population(
        pop_max=4, rng=rng, n_initial=4, task=TASK_L2V2,
        task_difficulty="balanced",
    )
    pop._record_logic_window(0, MODE_AND, True)
    assert pop._fitness_defined(0) is False
    pop._record_logic_window(0, MODE_NOT, True)
    assert pop._fitness_defined(0) is True


# ── ERRATA v3.3 (Path D1) — anti-follower structural seed for not_only ─────
def test_l2v2_not_only_seed_has_anti_follower_topology():
    """ERRATA v3.3 (D1): not_only founders must be born with anti-follower
    micro-circuit, not random noise.  Specifically:
      * A→A-detectors      strongly positive
      * A-detectors→output strongly NEGATIVE
      * S→S-tonic-drivers  positive
      * S-tonic→output     positive
    All other paths get small symmetric noise.
    """
    from archaea.neuron import N_HIDDEN, N_INPUT
    from archaea.population import L2V2_NOT_SEED_H_A_DET

    pop = Population(
        pop_max=64, rng=np.random.default_rng(11), n_initial=64,
        task=TASK_L2V2, task_difficulty="not_only",
    )
    w = pop.weights[pop.alive]              # (64, 220)
    w_ih = w[:, : N_INPUT * N_HIDDEN].reshape(-1, N_INPUT, N_HIDDEN)
    w_ho = w[:, N_INPUT * N_HIDDEN :].reshape(-1, N_HIDDEN)

    a_lo, a_hi = 0, N_CH_A
    s_lo, s_hi = N_CH_A + N_CH_B, N_INPUT
    h_det = L2V2_NOT_SEED_H_A_DET

    # A-detector pathway: A→hidden positive, hidden→output negative.
    a_to_det = w_ih[:, a_lo:a_hi, :h_det]
    det_to_out = w_ho[:, :h_det]
    assert a_to_det.mean() > 0.9, a_to_det.mean()
    assert (a_to_det > 0).mean() > 0.95, "A→A-det should be uniformly positive"
    assert det_to_out.mean() < -0.9, det_to_out.mean()
    assert (det_to_out < 0).mean() > 0.95, "A-det→out should be uniformly negative"

    # S-tonic pathway: S→hidden positive, hidden→output positive.
    s_to_tonic = w_ih[:, s_lo:s_hi, h_det:]
    tonic_to_out = w_ho[:, h_det:]
    assert s_to_tonic.mean() > 0.7, s_to_tonic.mean()
    assert (s_to_tonic > 0).mean() > 0.95, "S→S-tonic should be uniformly positive"
    assert tonic_to_out.mean() > 0.6, tonic_to_out.mean()
    assert (tonic_to_out > 0).mean() > 0.95, "S-tonic→out should be uniformly positive"

    # Background noise: everywhere else hovers near zero (within ±0.2).
    # Spot-check the B→A-detector cells (no structural role).
    b_to_det = w_ih[:, N_CH_A : N_CH_A + N_CH_B, :h_det]
    assert abs(b_to_det.mean()) < 0.05, b_to_det.mean()
    assert b_to_det.min() >= -0.21 and b_to_det.max() <= 0.21


def test_l2v2_not_only_seed_per_agent_diversity():
    """D1 must seed *structure*, not clones.  Each founder should have its
    own draw of noise — std across the population on any single weight cell
    must be > 0 so mutation+selection has variation to work on."""
    pop = Population(
        pop_max=128, rng=np.random.default_rng(20), n_initial=128,
        task=TASK_L2V2, task_difficulty="not_only",
    )
    w = pop.weights[pop.alive]
    # Per-cell std across the population: median should be clearly > 0.
    cell_std = w.std(axis=0)
    assert float(np.median(cell_std)) > 0.05, np.median(cell_std)
    # No two founders should be exactly identical.
    pair_diffs = (w[0:1] != w).any(axis=1).sum()
    assert pair_diffs >= 127, "expected unique founder weight vectors"


def test_l2v2_not_only_seed_produces_anti_following_behaviour():
    """Functional acceptance test: a freshly-seeded not_only colony should,
    on average, output near-silent when a=1 and spike when a=0 — *before
    any evolution has happened*.  This is the whole point of D1: the
    founder population is already a viable NOT specialist.
    """
    from archaea.neuron import NetworkBatch
    from archaea.oracle import (
        LOGIC_HIGH_HZ, LOGIC_LOW_HZ, S_NOT_HZ,
        OUT_SPIKING_THRESHOLD_HZ, poisson_three_channels,
    )

    rng = np.random.default_rng(2027)
    pop = Population(
        pop_max=64, rng=rng, n_initial=64,
        task=TASK_L2V2, task_difficulty="not_only",
    )
    weights = pop.weights[pop.alive].copy()

    # Drive the network for one 500 ms window with a=HIGH, b=LOW, S=NOT_HZ.
    # Target: silent (NOT 1 = 0).
    net_high = NetworkBatch(weights, rng=np.random.default_rng(101))
    spikes_high = poisson_three_channels(
        np.random.default_rng(102),
        f_a_hz=LOGIC_HIGH_HZ, f_b_hz=LOGIC_LOW_HZ, f_s_hz=S_NOT_HZ,
        duration_ms=500.0,
    )
    out_high = np.zeros(len(weights))
    for t in range(spikes_high.shape[0]):
        out_high += net_high.step(spikes_high[t]).reshape(-1)
    rate_high = out_high / 0.5  # Hz

    # Same colony, but a=LOW.  Target: spike (NOT 0 = 1).
    net_low = NetworkBatch(weights, rng=np.random.default_rng(103))
    spikes_low = poisson_three_channels(
        np.random.default_rng(104),
        f_a_hz=LOGIC_LOW_HZ, f_b_hz=LOGIC_LOW_HZ, f_s_hz=S_NOT_HZ,
        duration_ms=500.0,
    )
    out_low = np.zeros(len(weights))
    for t in range(spikes_low.shape[0]):
        out_low += net_low.step(spikes_low[t]).reshape(-1)
    rate_low = out_low / 0.5

    # Headline: median agent should spike on a=0 and stay below threshold on a=1.
    med_high = float(np.median(rate_high))
    med_low = float(np.median(rate_low))
    assert med_high < OUT_SPIKING_THRESHOLD_HZ, (
        f"founders should be quiet when a=high, got median {med_high:.1f} Hz"
    )
    assert med_low > OUT_SPIKING_THRESHOLD_HZ, (
        f"founders should spike when a=low, got median {med_low:.1f} Hz"
    )
    # Anti-correlation: a=low rate should be at least 2× a=high rate at the
    # population median — coarse but catches the topology being inverted.
    assert med_low > 2.0 * max(med_high, 1.0), (med_low, med_high)


def test_l2v2_not_only_seed_does_not_affect_mixed_dishes():
    """Default mixed dishes (balanced/uniform/extreme) must keep the
    Uniform(-0.5, +1.5) prior — the v2.5 anti-collapse contract.  D1 is
    *only* applied when task_difficulty == 'not_only'."""
    from archaea.population import L2V2_WEIGHT_INIT_LOW, L2V2_WEIGHT_INIT_HIGH

    for name in ("balanced", "uniform", "extreme", "and_only"):
        pop = Population(
            pop_max=64, rng=np.random.default_rng(hash(name) & 0xFFFF),
            n_initial=64, task=TASK_L2V2, task_difficulty=name,
        )
        mean_w = float(pop.weights[pop.alive].mean())
        assert 0.3 < mean_w < 0.7, (name, mean_w)
        lo = float(pop.weights[pop.alive].min())
        hi = float(pop.weights[pop.alive].max())
        assert lo >= L2V2_WEIGHT_INIT_LOW - 1e-9, (name, lo)
        assert hi <= L2V2_WEIGHT_INIT_HIGH + 1e-9, (name, hi)


# ── ERRATA v3.2 — specialist-dish wrong-answer penalty ─────────────────────
def test_specialist_presets_carry_negative_reward_wrong():
    """Both specialist dishes must ship a -BREATH_PER_WINDOW reward_wrong;
    mixed dishes must keep reward_wrong = 0 (v2.2 anti-collapse invariant)."""
    from archaea.economy import BREATH_PER_WINDOW
    for name in ("and_only", "not_only"):
        assert TASK_DIFFICULTY_PRESETS[name]["reward_wrong"] == pytest.approx(
            -BREATH_PER_WINDOW
        ), name
    for name in ("uniform", "balanced", "hard", "extreme"):
        assert TASK_DIFFICULTY_PRESETS[name]["reward_wrong"] == 0.0, name


def test_oracle_sample_applies_reward_wrong():
    """The reward_wrong arg must propagate to OracleSample.reward_wrong."""
    rng = np.random.default_rng(42)
    s = draw_oracle_sample(rng, reward_wrong=-1.25)
    assert s.reward_wrong == pytest.approx(-1.25)
    s_default = draw_oracle_sample(rng)
    assert s_default.reward_wrong == 0.0


def test_not_only_silent_strategy_has_negative_expected_credit():
    """Headline guarantee of v3.2: in not_only, an always-silent agent
    must have NEGATIVE expected credit per window so silent strategy
    starves out and selection pressure for true NOT can form.

    Computes expected per-window net credit analytically for the four
    archetypal strategies and asserts the ranking + signs."""
    from archaea.economy import BREATH_PER_WINDOW
    w = difficulty_weights("not_only")
    rwrong = w["reward_wrong"]

    # Half the windows have target=0 (a=1, agent should stay silent),
    # half have target=1 (a=0, agent should spike).  Compute expected
    # reward (table + effort bonus) per strategy.
    def strategy_net(spike_on_target_0: bool, spike_on_target_1: bool) -> float:
        # target=0 (silent is correct)
        if spike_on_target_0:
            r0 = rwrong + spike_effort_bonus(75.0, 0)         # 75 Hz = "spike"
        else:
            r0 = (R_NOT_SILENT_RAW * REWARD_SCALE) + spike_effort_bonus(0.0, 0)
        # target=1 (spike is correct)
        if spike_on_target_1:
            r1 = (R_NOT_SPIKE_RAW * REWARD_SCALE) + spike_effort_bonus(75.0, 1)
        else:
            r1 = rwrong + spike_effort_bonus(0.0, 1)
        return 0.5 * (r0 + r1) - BREATH_PER_WINDOW

    silent       = strategy_net(False, False)   # always silent
    always_spike = strategy_net(True,  True)
    smart_not    = strategy_net(False, True)    # silent when a=1, spike when a=0
    anti_not     = strategy_net(True,  False)   # spike when a=1, silent when a=0

    # The whole point of v3.2:
    assert silent < 0.0, f"silent must starve in not_only, got {silent}"
    # Always-spike still pays more than breath so day-zero excitatory founders
    # don't go extinct before mutations find smart NOT.
    assert always_spike > 0.0, f"always-spike must survive, got {always_spike}"
    # Smart NOT must out-reproduce always-spike (not necessarily by a huge
    # margin — selection just needs a ranking, not an arms race).
    assert smart_not > always_spike
    # Anti-NOT (perfectly wrong) is the worst of all.
    assert anti_not < silent


def test_specialist_silent_strictly_more_punished_than_mixed_silent():
    """v3.2 invariant: silent strategy must be much more punished in
    specialist dishes than in mixed dishes — that is the point of the
    new reward_wrong field.  Concretely, the specialist silent net must
    be at least one breath-cycle (1.25 credit/win) worse than the mixed
    silent net so the silent attractor in not_only / and_only collapses
    while the mixed-dish 'barely viable silent' contract is preserved.
    """
    from archaea.economy import BREATH_PER_WINDOW

    def silent_net(difficulty: str) -> float:
        w = difficulty_weights(difficulty)
        rwrong = w["reward_wrong"]
        p_and = w["p_mode_and"]
        p_a1 = w["p_and_target_one"]
        p_n1 = w["p_not_target_one"]
        # In each mode the silent agent's reward depends only on whether
        # this window's target_bit was 0 (silent correct) or 1 (silent wrong).
        # P(target=1 | AND) = p_a1; P(target=1 | NOT) = p_n1.
        and_silent_correct = R_AND_SILENT_RAW * REWARD_SCALE + spike_effort_bonus(0.0, 0)
        and_silent_wrong = rwrong + spike_effort_bonus(0.0, 1)
        not_silent_correct = R_NOT_SILENT_RAW * REWARD_SCALE + spike_effort_bonus(0.0, 0)
        not_silent_wrong = rwrong + spike_effort_bonus(0.0, 1)
        expected = (
            p_and * ((1 - p_a1) * and_silent_correct + p_a1 * and_silent_wrong)
            + (1 - p_and) * ((1 - p_n1) * not_silent_correct + p_n1 * not_silent_wrong)
        )
        return expected - BREATH_PER_WINDOW

    silent_balanced = silent_net("balanced")
    silent_not_only = silent_net("not_only")
    silent_and_only = silent_net("and_only")

    # Specialist silent must be DECISIVELY negative (well past the
    # "barely starving" range that mixed dishes hover in).
    assert silent_not_only < -0.5, silent_not_only
    assert silent_and_only < -0.5, silent_and_only
    # The gap between mixed and specialist must be at least
    # half a breath cycle — empirically it's exactly
    # |reward_wrong| × P(wrong window) = 1.25 × 0.5 = 0.625 for not_only.
    assert silent_balanced - silent_not_only > 0.4, (
        silent_balanced, silent_not_only
    )
    assert silent_balanced - silent_and_only > 0.4, (
        silent_balanced, silent_and_only
    )
    # Mixed dishes must still hover near zero (NOT punished hard) so the
    # v2.2 "soft anti-collapse" contract holds — within ±half a breath.
    assert abs(silent_balanced) < BREATH_PER_WINDOW / 2, silent_balanced


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


def test_l2v2_row_acc_isolates_silent_attractor():
    """The headline guarantee of v2.3.1: a forced-silent population must score
    0% on the (1,1) row even if its混合 acc_AND looks healthy.

    We do this by running a population in 'extreme' mode where 90% of AND
    questions expect target=1 — silent agents (output=0) get all of them
    wrong. acc_and_pop should plummet AND acc_and_11_pop must hit 0.
    """
    rng = np.random.default_rng(2027)
    pop = Population(
        pop_max=12, rng=rng, n_initial=12,
        task=TASK_L2V2, task_difficulty="extreme",
    )
    # Force every agent to be effectively silent: zero out every weight.
    pop.weights[:] = 0.0
    info = None
    for _ in range(80):
        info = pop.step_window()
    assert info is not None
    # Silent agents are 100% wrong on (1,1) — the row gauge MUST surface this
    # even if the user can't see it in the混合 AND temperature.
    assert info["row_acc"]["and_11"] < 0.05, (
        f"silent population scored {info['row_acc']['and_11']:.3f} on (1,1) — "
        "row telemetry isn't isolating the silent attractor"
    )
    # And NOT(0)=1 (silent's blind spot on NOT) should also be ~0
    assert info["row_acc"]["not_a0"] < 0.05


def test_l2v2_population_default_difficulty_balanced():
    """Default difficulty must be 'balanced' (the v2.3 50/50 setting) for backwards compat."""
    rng = np.random.default_rng(1)
    pop = Population(pop_max=4, rng=rng, n_initial=4, task=TASK_L2V2)
    assert pop.task_difficulty == "balanced"


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
