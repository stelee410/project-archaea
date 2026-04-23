"""SPEC_L2_V2.0 §2 — Pulsar input + Oracle Altar reward table.

Generates one 500 ms window's *(f_a, f_b, f_s, mode, target_bit, base_reward_table)*.
All agents in the population receive the same stimulus triple per window;
each agent is then judged independently against the truth table.

──────────────────────────────────────────────────────────────────────────
ERRATA v3.2 — Specialist-dish silent attractor: wrong-answer penalty
──────────────────────────────────────────────────────────────────────────
v3.1 enabled specialist cultivation but real runs of `not_only` still
collapsed into a 100%-silent population (700 alive / 0 deaths after
40+ minutes; (a=0, target=1) row stayed at 0%).  Diagnosis:

The v2.2 reward table was tuned for the *mixed* dish — it deliberately
keeps silent's expected income just above breath ("soft anti-collapse")
so a fresh population doesn't go extinct before AND/NOT learners emerge.
In `not_only` that arithmetic flips into a death trap for evolution:

    silent net /win = 0.5 × (R_NOT_SILENT × scale + effort − BREATH)
                    + 0.5 × (0 + 0 − BREATH)
                    = 0.5 × (2.5 + 0.1 − 1.25) + 0.5 × (−1.25)
                    = +0.05/win  ← silent is profitable

silent agents never starve → no metabolic selection pressure → mutants
that try to spike on (a=0, target=1) die from "spike on a=1, wrong"
windows about as fast as they reproduce → NOT logic cannot emerge.

Fix: in specialist dishes only (and_only / not_only), set
``reward_wrong = -BREATH_PER_WINDOW = -1.25``.  This makes wrong answers
exactly cancel one window's metabolic cost — so silent_on_wrong becomes
−2.5/win, dragging silent's per-window net to:

    not_only silent net = (−2.5 + 1.35) / 2 = −0.575/win
                          → starves in ~87 windows (44 s)
    not_only always-spike net = (+23.85 + −2.5) / 2 = +10.675/win
                          → still viable (excitatory founders survive)
    not_only smart NOT net = (+23.85 + 1.35) / 2 = +12.6/win
                          → 18% reproduction-speed advantage over
                            always-spike  → genuine selection forms

Mixed dishes keep ``reward_wrong = 0`` — the v2.2 anti-collapse contract
holds (silent net hovers near zero across uniform/balanced/hard/extreme,
within ±0.5 breath; specialist silent is decisively below by ≥0.4 credit/
win, the gap the v3.2 regression test pins).

Implementation: ``TASK_DIFFICULTY_PRESETS`` grew a fourth field
``reward_wrong``; ``draw_oracle_sample`` accepts it as a kwarg;
``Population.step_window`` reads it from the difficulty preset and
threads it through.  See docs/project-summary.md §5.10.

──────────────────────────────────────────────────────────────────────────
ERRATA v3.1 — Specialist-dish fixes (admixture cultivation prerequisites)
──────────────────────────────────────────────────────────────────────────
Two bugs only show up in the and_only / not_only specialist dishes added
for the SPEC_L2_V3.0 admixture mixer:

    (A) ``Population._fitness_defined`` required samples in BOTH the AND
        buffer and the NOT buffer.  In a specialist dish one of them is
        permanently empty, so every agent stayed "fitness undefined" for
        the entire run — global_sigma never annealed (locked at SIGMA_BASE)
        and elite-victim selection degraded to credit-only.  Fixed by
        deriving _needs_and_samples / _needs_not_samples from the difficulty
        preset's p_mode_and and gating fitness on only the modes the
        environment actually produces.

    (B) The v2.5 founder distribution Uniform(-0.5, +1.5) is excitation-
        biased so day-zero founders emit some output.  This is anti-
        aligned with NOT logic, so v3.1 had not_only mirror it to
        Uniform(-1.5, +0.5).  **SUPERSEDED in v3.3**: that mirror, paired
        with v3.2's silent-agent starvation penalty, killed the entire
        founder population in the first few windows (uniformly inhibitory
        weights ⇒ silent ⇒ starve before they can reproduce).  v3.3
        replaces (B) with a structural anti-follower seed (Path D1); see
        the v3.3 block below.

The (A) fix from v3.1 stays.  Mixed dishes (uniform / balanced / hard /
extreme) and the and_only dish keep the v2.5 excitatory founders and the
"both modes required" fitness contract.  See docs/project-summary.md §5.9.

──────────────────────────────────────────────────────────────────────────
ERRATA v3.3 — Path D1: anti-follower structural seed for not_only
──────────────────────────────────────────────────────────────────────────
v3.1's "uniformly-inhibitory founder" was a misdiagnosis.  NOT logic
isn't "negative weights everywhere"; it's *context-gated routing* —
"A high → suppress output, A low → let a tonic source fire output".
A Uniform(-1.5, +0.5) prior makes every founder silent (E[Σw·s] < 0 ⇒
I_o below threshold) and v3.2's wrong-answer penalty starves them before
mutation can stumble onto the routing topology.  Diagnosis: random or
uniformly-inhibitory founders cannot find a context-gated micro-circuit
inside the time-budget the colony runs at; the search space is too large.

v3.3 replaces noise with structure.  Each not_only founder is born with
a hand-designed anti-follower micro-circuit:

    Hidden split into two functional groups:
      A-detectors (cols 0..H_A_DET-1):
          A→hidden  : strong positive  → tracks A channel
          hidden→out: strong NEGATIVE  → A high suppresses output
      S-tonic drivers (cols H_A_DET..N_HIDDEN-1):
          S→hidden  : positive         → S=80Hz keeps these firing
          hidden→out: positive         → drives output unless suppressed

Behaviour at S=80Hz (NOT mode):
    a=1 (75Hz) → A-detectors fire   → suppress output  → silent  ✓
    a=0 (25Hz) → A-detectors quiet  → S-tonic drives   → spike   ✓

Each founder draws independent noise within the structural bands so
mutation+selection still operates on real variation; what's removed is
the *abiogenesis* burden of inventing the routing pattern from random
weights.  Biological precedent: GABAergic neurons evolved from
glutamatergic ancestors via single postsynaptic ion-channel polarity
flip; ON/OFF retinal ganglion cells share architecture but differ at
one synapse's polarity.  The constants live in archaea/population.py
near ``L2V2_NOT_SEED_*``; full rationale in
docs/project-summary.md §5.11.

──────────────────────────────────────────────────────────────────────────
ERRATA v2.5 — Prebiotic-stage founder bias (companion to v2.4)
──────────────────────────────────────────────────────────────────────────
Diagnosis after deploying v2.4 to a fresh seed: the colony died at
t_sim ≈ 311 s with row_acc.and_11 = 0 / row_acc.not_a0 = 0 — i.e. NO
agent in 622 windows ever crossed the spike threshold.  v2.4's threshold
drop (50→20 Hz) and effort bonus only help individuals that *already*
emit something; they cannot rescue a population whose founders all sit
at f_out = 0 because Uniform(-1.5, 1.5) gives Σw ≈ 0 ⇒ I_o ≈ 0.

The deadlock: silent → net credit < 0 → never reaches C_REPRO=200 →
never reproduces → never mutates → never escapes silence → starves
in ~333 s.  This is not a fitness-landscape problem; it is an
*evolvability=0* problem — the evolutionary loop simply never starts.

v2.5 fixes the founder distribution itself: Uniform(-0.5, 1.5) instead
of (-1.5, 1.5).  This shifts E[Σw] positive so a useful fraction of
day-zero founders emit f_out > 0, lighting the fuse on the variation-
selection-reproduction cycle.  Negative weights still cover ~25% of the
range — well above SPEC §3.1's "≥20% negative" requirement.

The actual implementation lives in archaea/population.py near
``L2V2_WEIGHT_INIT_LOW``; see that comment block for the full biological
framing (LTEE-style "prebiotic selection", not intelligent design — we
study evolution-of-logic from a founder that already produces output,
not abiogenesis from random monomers).

──────────────────────────────────────────────────────────────────────────
ERRATA v2.4 — Platform-cliff fix: lower spike threshold + effort gradient
──────────────────────────────────────────────────────────────────────────
v2.3 environment-shaping (50% target=1) removed the 75% silent ceiling but
exposed a deeper problem: the *fitness landscape* was a "platform-cliff".
Diagnosis of a long-running v2.3 colony showed the reigning "best" agent
emitted f_out ≈ 26-32 Hz on (1,1) AND windows — clearly attempting to
spike — yet was classified as a "silent" 0 because OUT_SPIKING_THRESHOLD_HZ
was set to 50 Hz (halfway between LOGIC_LOW and LOGIC_HIGH).  This means:

    * any agent whose mutations give it nascent spiking behaviour (10-49 Hz
      output) pays BREATH_PER_WINDOW for the spikes but receives ZERO extra
      reward — same outcome as fully silent.
    * the only payoff cliff is at 50 Hz, but reaching 50 Hz from 0 Hz
      requires *several* coordinated weight mutations with no intermediate
      reward.  Evolution has no upward gradient on the platform.

Biological analogy: half-an-eye is useless if the brain only counts photons
above the cone-cell saturation threshold.  Real biology *does* reward
nascent function — that's what gradualism (Darwin) requires.

v2.4 makes two changes (named A + C in the design discussion):

    (A) OUT_SPIKING_THRESHOLD_HZ : 50.0 → 20.0
        Lowers the "judged as 1" threshold so an agent already attempting
        to spike (the 26-32 Hz cohort observed in v2.3) is recognised by
        the oracle as a correct '1' and immediately picks up R_AND_SPIKE.
        This is *environment shaping*, not a reward change — analogous to
        a niche where any photon detection (not just saturation) helps.

    (C) Spike-effort micro-bonus: spike_effort_bonus(f_out, target_bit)
        Adds a small reward gradient ∈ [0, SPIKE_EFFORT_BONUS_K] for output
        rates that move *in the correct direction* for the current target
        bit.  Critically, the bonus is direction-gated:
            target=1: bonus = K · min(1, f_out / threshold)
                      → silent in (1,1) gets ~0; near-spike (26 Hz) gets ~0.65K
            target=0: bonus = K · (1 - min(1, f_out / threshold))
                      → silent in (0,0) gets full K; spurious spike gets ~0
        This preserves SPEC's "oracle-only judgement" principle (no per-
        synapse online learning) while giving the *population* a smooth
        upward gradient instead of a cliff.  Silent agents still earn
        almost exactly what they did under v2.3 (since they win on most
        target=0 windows by accident anyway), but a 26 Hz attempt on (1,1)
        now nets a small positive credit instead of zero — preserving the
        mutation lineage long enough for it to evolve to >threshold.

K is small (0.1) by design — the goal is to keep the gradient *visible*
without distorting the spike-vs-silent reward ratio.  Per-window expected
income changes:

    silent (v2.3 → v2.4):  net −0.125/win → net ~−0.075/win  (still starves slowly)
    perfect (v2.3 → v2.4): net +9.875/win → net +9.975/win   (essentially unchanged)
    near-spike (26 Hz on (1,1)):
        v2.3: 0 credit on (1,1) win + breath cost = net negative
        v2.4: 0 base + 0.13 effort bonus + 3.75 from (A) classifying
              26 Hz > 20 Hz as '1' → R_AND_SPIKE_CORRECT applies → net positive

The (A) reclassification carries the bulk of the fitness change; (C) is
the safety net for sub-threshold attempts.

──────────────────────────────────────────────────────────────────────────
ERRATA v2.3 — Weighted environment (target=1 occurs 50%, was 37.5%)
──────────────────────────────────────────────────────────────────────────
v2.2 kept *uniform* sampling over (mode, A, B) — i.e. each of the 4 AND
truth-table rows + 2 NOT rows had equal probability.  This produces a
target=1 prior of only 0.5×0.25 + 0.5×0.5 = **37.5%**, and an acc_AND
ceiling for silent agents of **75%** (3/4 AND rows expect 0).  An 18-min
run on this distribution stalled at acc_AND ≈ 73% — bumping into the
silent ceiling, indistinguishable on the dashboard from genuine learning.

v2.3 changes the *environment*, not the rewards: we sample so target=1
occurs 50% of the time:

    P(MODE_AND) = 0.5
        within AND:  P(A=1, B=1)  = 0.50    target=1   ← upweighted ×2
                     P(A=0, B=0)  = 0.50/3  target=0
                     P(A=0, B=1)  = 0.50/3  target=0
                     P(A=1, B=0)  = 0.50/3  target=0
    P(MODE_NOT) = 0.5
        within NOT:  P(A=0)       = 0.50    target=1
                     P(A=1)       = 0.50    target=0   (B uniform; distractor)

This drops the silent ceiling to:
    silent acc_AND = 0.5 (was 0.75)   ← 75% plateau is gone
    silent acc_NOT = 0.5 (unchanged)

Combined with v2.2 reward table:
    silent  per-window = 0.5·0.5·2.0 + 0.5·0.5·2.5 = 1.125  (net −0.125/win)
    perfect per-window = 0.5·(0.5·2.0+0.5·15) + 0.5·(0.5·2.5+0.5·25)
                       = 11.125  (net +9.875/win, ~10 s/repro)
    ratio: 9.9× per-window, ~80× reproduction-rate

silent agents now slowly starve (~3 min half-life from credit 50) instead
of stalling at the 75% plateau.  Founder-collapse risk is gone because
the population already has ≥10% true-logic elites at the v2.3 switch.

──────────────────────────────────────────────────────────────────────────
ERRATA v2.2 — "Soft" anti silent-collapse (revised from v2.1)
──────────────────────────────────────────────────────────────────────────
Background — the silent attractor (still relevant):

The original SPEC §2.2 raw table (+20 / +5 / +50 / +10) suffered a
class-imbalance bias: 3 of the 4 AND truth-table rows want target=0
(silent), and 1 of the 2 NOT rows wants target=0.  A trivially "always
silent" agent therefore picks up acc_AND ≈ 75% / acc_NOT ≈ 50% and the
dashboard *looks like* the swarm learned the logic — when in fact it
just learned to avoid spiking.  The genuine logic answers (1 AND 1 → 1,
NOT 0 → 1) require coordinated mutations including inhibitory paths and
were never reached under SPEC's narrow reward gap.

v2.1 attempt and failure:

v2.1 tried to break the attractor by pushing silent rewards BELOW breath
(0.5 / 1.0 vs breath 1.25), making "always silent" net −0.81/window.
Mathematically clean — but ecologically catastrophic.  Initial agents
have random weights with ~0 expected output current (SPEC §3.1 forces
≥20% negative weights, so Σw ≈ 0 ⇒ I_o ≈ 0 ⇒ f_out ≈ 0); virtually all
of them ARE silent agents at birth.  v2.1 starved the entire founding
generation in 60 windows (~30 s wall-clock at 20 Hz sim).  Mutation
needs reproduction (C_REPRO=200), but a starving population never
reproduces, so v2.1 produced 100% extinction with no chance to discover
the logic.

v2.2 design — silent must be VIABLE, not lethal:

Keep the spike-correct premiums (15 / 25 scaled) so any agent that
discovers the right answer暴富.  Set silent rewards *just above* breath
so silent agents survive and reproduce slowly — buying mutation enough
generations to find the spike-correct answers, while still being
crushed by the 17× faster reproduction rate of true-logic agents.

    silent:    AND silent · 2.0 · 3/4  +  NOT silent · 2.5 · 1/2  = 1.375 / win
                                                                  net +0.125 → ~400 win/repro
    perfect:   AND mode  · (2.0·3/4 + 15·1/4) + NOT mode · (2.5·1/2 + 25·1/2)
                                                                  = 9.50 / win
                                                                  net +8.25 → ~24 win/repro
    ratio:     6.9× per-window reward, 17× reproduction rate

Old vs new scaled rewards:

    | mode | target | SPEC scaled | v2.1 scaled | v2.2 scaled | rationale            |
    | AND  | 1      | 5.00        | 15.00       | 15.00       | spike premium ×3     |
    | AND  | 0      | 1.25        |  0.50       |  2.00       | viable silent生存    |
    | NOT  | 1      | 12.50       | 25.00       | 25.00       | high-difficulty ×2   |
    | NOT  | 0      | 2.50        |  1.00       |  2.50       | viable silent生存    |

REWARD_SCALE stays at 0.25.  Tests pin the v2.2 invariants:
   silent_per_window > BREATH_PER_WINDOW   (no extinction)
   perfect_per_window > 5 × silent          (clear gradient)

──────────────────────────────────────────────────────────────────────────

Interpretation note (off-SPEC clarification, see SPEC_L2_V2.0 §3.2):
  the SPEC mentions an "eligibility trace" for the hidden layer.  In
  Archaea the only legal long-term storage is mutation-on-birth (SPEC
  §11; "no STDP, no Hebbian, no eligibility traces").  We therefore
  implement the **weak version**: the reward table itself is the credit
  assignment — agents whose hidden layer is wired to compute the right
  function for the current S receive Credit, reproduce, and pass weights
  on.  Per-synapse online learning is intentionally NOT done.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .economy import BREATH_PER_WINDOW
from .neuron import N_INPUT
from .stimulus import poisson_spikes_window
from .task import N_CH_A, N_CH_B, N_CH_S

# ── Channel encoding ────────────────────────────────────────────────────────
LOGIC_LOW_HZ = 25.0   # < 50  → bit 0
LOGIC_HIGH_HZ = 75.0  # > 50  → bit 1
BIT_THRESHOLD_HZ = 50.0  # f > 50 ⇒ True (SPEC §2.1)

# Selector channel S — instruction frequencies (SPEC §2.1)
S_AND_HZ = 20.0
S_NOT_HZ = 80.0
S_JITTER_HZ = 5.0  # ±5Hz jitter as per SPEC

# Output classification — agent emits "Spiking" iff f_out > this threshold.
# v2.4 lowered from 50.0 → 20.0 to fix the platform-cliff fitness landscape
# (see ERRATA v2.4 in module docstring): the v2.3 colony had a cohort of
# near-spiking agents (~26-32 Hz on (1,1)) that were classified as silent
# because 50 Hz was unreachable from 0 Hz in a single mutation.  20 Hz is
# still ~3× LOGIC_LOW_HZ so spurious noise rarely crosses it, but nascent
# spike attempts now get correctly rewarded.  Tunable; not SPEC-fixed.
OUT_SPIKING_THRESHOLD_HZ = 20.0

# v2.4 spike-effort micro-bonus coefficient (see ERRATA v2.4).  Adds a
# directionally-correct gradient ∈ [0, K] on top of the table reward,
# guaranteeing that "trying to spike on a target=1 window" pays slightly
# more than "fully silent on a target=1 window" even before the agent
# reaches OUT_SPIKING_THRESHOLD_HZ.  Kept small (0.1) so the table reward
# remains the dominant evolutionary force.
SPIKE_EFFORT_BONUS_K = 0.1

# ── Mode identifiers ────────────────────────────────────────────────────────
MODE_AND = 0
MODE_NOT = 1
MODE_NAMES = {MODE_AND: "AND", MODE_NOT: "NOT"}

# ── Reward table (v2.2 — see ERRATA in module docstring) ─────────────────
#
# Scaled values (× REWARD_SCALE) are what actually arrives at credit:
#     AND spike  = 15.0     AND silent = 2.0
#     NOT spike  = 25.0     NOT silent = 2.5
#
# Soft anti-collapse design:
#   - silent rewards (2.0 / 2.5) keep silent expected income (1.375/win)
#     marginally above BREATH_PER_WINDOW=1.25 — silent agents survive and
#     can slowly reproduce, giving mutation generations to explore.
#   - spike rewards (15 / 25) are massively higher per-event so any agent
#     discovering "1 AND 1 → spike" or "NOT 0 → spike" reproduces ~17×
#     faster than the silent baseline and quickly dominates.
REWARD_SCALE = 0.25
R_AND_SPIKE_RAW = 60.0    # was 20  — 3× to overpower silent attractor on AND(1,1)
R_AND_SILENT_RAW = 8.0    # was  5  — slightly above breath to keep silent viable
R_NOT_SPIKE_RAW = 100.0   # was 50  — high-difficulty premium remains highest
R_NOT_SILENT_RAW = 10.0   # was 10  — SPEC §2.2 original (already viable on its own)


# ── Sampling weights (v2.3 — see ERRATA in module docstring) ─────────────
# Environment-shaping: target=1 should occur 50% of the time so that a
# permanent-silence strategy maxes out at acc_AND=0.5 instead of 0.75.
# Tunable; tests pin the marginal P(target=1)=0.5 invariant.
P_MODE_AND = 0.5                       # P(MODE_AND); P(MODE_NOT) = 1 - this
P_AND_TARGET_ONE = 0.5                 # P(A=1, B=1 | AND)
P_NOT_TARGET_ONE = 0.5                 # P(A=0     | NOT)


# ── Task-difficulty presets (v2.3 environment slider) ────────────────────
# Each preset shapes the environment WITHOUT touching the reward table.
# The user picks one at colony launch ("上帝塑造生态"); the population must
# adapt or starve ("古菌自己活下来").
#
#   uniform : SPEC §2.2 original — 25% target=1 on AND rows.
#             silent ceiling: acc_AND ≈ 75% (looks like learning, isn't).
#   balanced: v2.3 default — 50% target=1 on both modes.
#             silent ceiling: acc_AND ≈ 50% (lazy strategy is exposed).
#   hard    : 70% — strong selection toward true logic; founders may struggle.
#   extreme : 90% — only true-logic agents survive; expect mass extinction
#             unless the population already has elite members.
#
# SPEC_L2_V3.0 §2.4 — single-gate specialist presets for the admixture
# experiment: cultivate AND-experts in one dish, NOT-experts in another, then
# 💾 save each as a Strain and ⚗️ Mixer them together.  These intentionally
# only ask one kind of question so the population can over-fit to it; the
# fitness function then naturally rewards "really learning" rather than
# "balancing two skills".
#
#   and_only: P(AND)=1.  All windows ask AND; NOT machinery is irrelevant
#             and may even be selected against.  Use to build AND-学家.
#   not_only: P(AND)=0.  All windows ask NOT.  Use to build NOT-学家.
#
# Within each specialist, target=1 sampling is held at 50% so the silent
# attractor stays at the v2.3 ceiling (50%) — same as `balanced`.
#
# ERRATA v3.2 — wrong-answer penalty in specialist dishes.
# ───────────────────────────────────────────────────────────
# v2.2's reward table was tuned for the *mixed* dish: silent earns
# +0.125/win across AND+NOT (just above breath, "soft anti-collapse").
# In `not_only` that arithmetic flips: silent on (a=1, target=0) gets
# R_NOT_SILENT × 0.25 = 2.5 → +1.35/win, but silent on (a=0, target=1)
# only loses −1.25/win (breath only — reward_wrong=0).  Mean = +0.05/win,
# which is metabolically *positive* — silent agents never starve, so no
# selection pressure ever forms and NOT logic cannot emerge.  Diagnostic
# symptom: 700 alive / 0 deaths / 0% on the (a=0, target=1) row after
# 40+ minutes of `not_only` simulation.
#
# Fix: in specialist dishes, set reward_wrong = -BREATH_PER_WINDOW so
# wrong answers exactly cancel one window's metabolic cost.  Then:
#   not_only silent net = (−2.5 + 1.35) / 2 = −0.575/win  → dies in ~87
#                                                          windows (44 s)
#   not_only always-spike net = (+23.85 + −2.5) / 2 = +10.675/win  → lives
#   not_only smart NOT net    = (+23.85 + 1.35) / 2 = +12.6/win    → 18%
#       reproduction-speed advantage over always-spike → genuine selection
#
# Mixed dishes keep reward_wrong=0 (the v2.2 invariant: silent must remain
# barely viable in mixed environments so the population doesn't go extinct
# before the first AND-or-NOT learner emerges).
TASK_DIFFICULTY_PRESETS: dict[str, dict[str, float]] = {
    "uniform":  {"p_mode_and": 0.5, "p_and_target_one": 0.25, "p_not_target_one": 0.5, "reward_wrong": 0.0},
    "balanced": {"p_mode_and": 0.5, "p_and_target_one": 0.50, "p_not_target_one": 0.5, "reward_wrong": 0.0},
    "hard":     {"p_mode_and": 0.5, "p_and_target_one": 0.70, "p_not_target_one": 0.7, "reward_wrong": 0.0},
    "extreme":  {"p_mode_and": 0.5, "p_and_target_one": 0.90, "p_not_target_one": 0.9, "reward_wrong": 0.0},
    # SPEC_L2_V3.0 specialist dishes — wrong-answer penalty turns the silent
    # attractor from "free retirement" into "starve in 44 s" so selection
    # pressure for true logic actually forms.  See ERRATA v3.2 above.
    "and_only": {"p_mode_and": 1.0, "p_and_target_one": 0.50, "p_not_target_one": 0.5, "reward_wrong": -BREATH_PER_WINDOW},
    "not_only": {"p_mode_and": 0.0, "p_and_target_one": 0.50, "p_not_target_one": 0.5, "reward_wrong": -BREATH_PER_WINDOW},
}
DEFAULT_TASK_DIFFICULTY = "balanced"


def difficulty_weights(name: str) -> dict[str, float]:
    """Resolve a difficulty preset name to its sampling-weight dict.

    Raises ValueError for unknown names.  Keep this the *only* place that
    maps preset → numeric weights so the SetupPage UI and the simulation
    backend can never disagree.
    """
    if name not in TASK_DIFFICULTY_PRESETS:
        raise ValueError(
            f"unknown task_difficulty {name!r}; "
            f"valid: {sorted(TASK_DIFFICULTY_PRESETS)}"
        )
    return dict(TASK_DIFFICULTY_PRESETS[name])  # copy so callers can't mutate the registry


@dataclass
class OracleSample:
    """One 500 ms window's stimulus + ground truth."""

    mode: int                       # MODE_AND or MODE_NOT
    bit_a: int                      # 0 or 1
    bit_b: int                      # 0 or 1   (always sampled, even in NOT mode)
    f_a_hz: float
    f_b_hz: float
    f_s_hz: float
    target_bit: int                 # expected output bit
    reward_correct: float           # Credit awarded if agent classifies correctly
    reward_wrong: float = 0.0       # Credit awarded if agent classifies wrong

    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self.mode]


def _reward_correct(mode: int, target_bit: int) -> float:
    """Oracle Altar reward — see ERRATA v2.2 in module docstring.

    Spike-correct (target_bit=1) values are amplified vs. SPEC §2.2 to
    overpower the "always silent" attractor.  Silent-correct values stay
    just above BREATH_PER_WINDOW so silent agents survive (giving
    mutation generations to explore) but reproduce 17× slower than
    true-logic agents.
    """
    if mode == MODE_AND:
        return (R_AND_SPIKE_RAW if target_bit == 1 else R_AND_SILENT_RAW) * REWARD_SCALE
    return (R_NOT_SPIKE_RAW if target_bit == 1 else R_NOT_SILENT_RAW) * REWARD_SCALE


def draw_oracle_sample(
    rng: np.random.Generator,
    p_mode_and: float | None = None,
    p_and_target_one: float | None = None,
    p_not_target_one: float | None = None,
    reward_wrong: float = 0.0,
) -> OracleSample:
    """Draw one 500 ms stimulus triple and the matching ground truth.

    v2.3 weighted sampling (see ERRATA in module docstring): the four AND
    rows are NOT uniform — (1,1) gets P=p_and_target_one, the three
    target=0 rows share the rest equally.  NOT-mode A is drawn so that
    target=1 (A=0) occurs with P=p_not_target_one; B is uniform
    (distractor).

    Defaults (None) fall back to module constants P_MODE_AND /
    P_AND_TARGET_ONE / P_NOT_TARGET_ONE — the v2.3 50/50 setup.

    Pass explicit values to *shape the environment* (the "上帝" knob —
    SPEC §0 difficulty preset).  Higher p_*_target_one ⇒ lazy "always
    silent" strategy is exposed faster, but the population must adapt
    or starve.  This is the design space behind the difficulty slider.

    ``reward_wrong`` (ERRATA v3.2): credit awarded when the agent's
    classification is wrong.  Defaults to 0 (the v2.2 mixed-dish
    invariant — wrong answers cost only breath, no extra punishment, so
    new agents have time to learn).  Specialist dishes pass
    ``-BREATH_PER_WINDOW`` to cancel the silent-correct windfall and
    force selection pressure to form.
    """
    p_m_and = P_MODE_AND if p_mode_and is None else float(p_mode_and)
    p_a1 = P_AND_TARGET_ONE if p_and_target_one is None else float(p_and_target_one)
    p_n1 = P_NOT_TARGET_ONE if p_not_target_one is None else float(p_not_target_one)

    mode = MODE_AND if rng.random() < p_m_and else MODE_NOT
    if mode == MODE_AND:
        if rng.random() < p_a1:
            bit_a, bit_b = 1, 1                              # the only target=1 AND row
        else:
            # 3 target=0 rows: (0,0) / (0,1) / (1,0), equally weighted
            row = int(rng.integers(0, 3))
            bit_a, bit_b = [(0, 0), (0, 1), (1, 0)][row]
        target_bit = 1 if (bit_a == 1 and bit_b == 1) else 0
        s_centre = S_AND_HZ
    else:
        # NOT operates on A; weight A so target=1 (a=0) gets the chosen probability.
        # B remains a uniform distractor — still binary, still drives the network.
        bit_a = 0 if rng.random() < p_n1 else 1
        bit_b = int(rng.integers(0, 2))
        target_bit = 1 - bit_a
        s_centre = S_NOT_HZ
    f_s = float(s_centre + rng.uniform(-S_JITTER_HZ, S_JITTER_HZ))
    f_a = LOGIC_HIGH_HZ if bit_a else LOGIC_LOW_HZ
    f_b = LOGIC_HIGH_HZ if bit_b else LOGIC_LOW_HZ
    rwd = _reward_correct(mode, target_bit)
    return OracleSample(
        mode=mode,
        bit_a=bit_a,
        bit_b=bit_b,
        f_a_hz=f_a,
        f_b_hz=f_b,
        f_s_hz=f_s,
        target_bit=target_bit,
        reward_correct=rwd,
        reward_wrong=float(reward_wrong),
    )


def poisson_three_channels(
    rng: np.random.Generator,
    f_a_hz: float,
    f_b_hz: float,
    f_s_hz: float,
    duration_ms: float,
) -> np.ndarray:
    """Generate (n_steps, N_INPUT) Poisson spikes split A=4, B=4, S=2.

    Concatenation order is fixed:  [A | B | S]  along the neuron axis.
    """
    if N_CH_A + N_CH_B + N_CH_S != N_INPUT:
        raise ValueError(
            f"channel split {N_CH_A}+{N_CH_B}+{N_CH_S} != N_INPUT={N_INPUT}"
        )
    sa = poisson_spikes_window(rng, float(f_a_hz), duration_ms, N_CH_A)
    sb = poisson_spikes_window(rng, float(f_b_hz), duration_ms, N_CH_B)
    ss = poisson_spikes_window(rng, float(f_s_hz), duration_ms, N_CH_S)
    return np.concatenate([sa, sb, ss], axis=1)


def classify_output(f_out_hz: float) -> int:
    """Agent's binary verdict for one window."""
    return 1 if float(f_out_hz) > OUT_SPIKING_THRESHOLD_HZ else 0


def spike_effort_bonus(f_out_hz: float, target_bit: int) -> float:
    """Directionally-correct spike-effort subsidy (ERRATA v2.4, design "C").

    Returns a small reward ∈ [0, SPIKE_EFFORT_BONUS_K] that is added on top
    of the oracle table reward.  The bonus is shaped so it only rewards
    *effort in the correct direction* for the current ground-truth bit:

        target_bit == 1 (the agent SHOULD spike):
            bonus = K · min(1, f_out / OUT_SPIKING_THRESHOLD_HZ)
            silent f_out=0   → 0       (no payoff for laziness)
            near-spike 16 Hz → 0.8 K   (rewards approaching the threshold)
            full spike       → K       (saturates at threshold; no double-pay
                                        with the big spike-correct table reward)

        target_bit == 0 (the agent SHOULD stay silent):
            bonus = K · (1 - min(1, f_out / OUT_SPIKING_THRESHOLD_HZ))
            silent f_out=0   → K       (rewards correct silence)
            spurious spike   → 0       (no payoff for incorrect spiking)

    Together with the lowered OUT_SPIKING_THRESHOLD_HZ (design "A"), this
    smooths the fitness landscape into an upward gradient — agents whose
    mutations push them toward the correct output for the current task
    (regardless of which task the current window happens to be) gain a
    small positive credit signal that accumulates over many windows.

    Per the SPEC §11 ban on per-synapse online learning this is *still
    table-driven* — it's a richer table indexed on (f_out, target_bit)
    rather than just (mode, target_bit), evaluated once per agent per
    window and applied to the credit balance just like the base reward.
    """
    norm = min(1.0, max(0.0, float(f_out_hz)) / OUT_SPIKING_THRESHOLD_HZ)
    if int(target_bit) == 1:
        return SPIKE_EFFORT_BONUS_K * norm
    return SPIKE_EFFORT_BONUS_K * (1.0 - norm)
