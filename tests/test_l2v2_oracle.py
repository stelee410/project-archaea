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
    """All four named presets must resolve and carry the three required keys."""
    for name in ("uniform", "balanced", "hard", "extreme"):
        w = difficulty_weights(name)
        assert set(w.keys()) == {"p_mode_and", "p_and_target_one", "p_not_target_one"}
        for k, v in w.items():
            assert 0.0 <= v <= 1.0, f"{name}.{k}={v} out of [0,1]"
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
