"""SPEC_L2_V2.0 §2 — Pulsar input + Oracle Altar reward table.

Generates one 500 ms window's *(f_a, f_b, f_s, mode, target_bit, base_reward_table)*.
All agents in the population receive the same stimulus triple per window;
each agent is then judged independently against the truth table.

──────────────────────────────────────────────────────────────────────────
ERRATA v2.1 — Anti "silent collapse" rebalance (off-SPEC, deliberate)
──────────────────────────────────────────────────────────────────────────
The original SPEC §2.2 raw table (+20 / +5 / +50 / +10 — see commit
history) suffered a class-imbalance attractor: 3 of the 4 AND truth-table
rows want target=0 (silent), and 1 of the 2 NOT rows wants target=0.
A trivially "always silent" agent therefore picks up:

    AND mode silent (3 of 4)  · 5  · 0.25 = 0.94 / window  (75% acc_AND)
    NOT mode silent (1 of 2)  · 10 · 0.25 = 1.25 / window  (50% acc_NOT)
                                                ─────
    average                                     ≈ 1.09 / window

against breath = 1.25/window (economy.py).  That's a roughly break-even
strategy that lets a never-spiking agent survive.  Worse, the population
average displays acc_AND ≈ 75% on the dashboard, which **looks like the
swarm has learned AND** when in fact it has learned to avoid spiking.

The actual logic — `1 AND 1 → spike`, `NOT 0 → spike` — is locally costly
to discover (needs coordinated mutations across the 220 weights including
inhibitory paths for NOT).  Without a strong gradient pulling toward those
two answers, mutation-only evolution settles into the silent attractor.

This rebalance preserves SPEC §2.2 *intent* (NOT(0) is the high-difficulty
premium, AND(1,1) is the only AND-spike answer) but widens the
spike-correct vs. silent-correct gap so:

    silent:    AND silent · 0.5 · 3/4  +  NOT silent · 1.0 · 1/2  = 0.44 / win
    perfect:   AND mode  · (0.5·3/4 + 15·1/4) + NOT mode · (1·1/2 + 25·1/2)
                                                                = 8.56 / win

  → "always silent" now nets −0.81 / window (DIES in ~60 windows from
    starting credit 50)
  → "perfect logic" nets +7.31 / window (REPRODUCES in ~27 windows)
  → ratio 19× (was 4.4×)

Old vs. new scaled rewards:

    | mode | target | OLD scaled | NEW scaled | rationale                  |
    | AND  | 1      | 5.00       | 15.00      | 3× — break silent floor    |
    | AND  | 0      | 1.25       | 0.50       | minimal生存, can't sustain |
    | NOT  | 1      | 12.50      | 25.00      | 2× — keep highest premium  |
    | NOT  | 0      | 2.50       | 1.00       | minimal生存                |

REWARD_SCALE stays at 0.25 — the "raw" SPEC-style numbers below
(R_*_RAW) are written out so the absolute economy magnitudes remain
clear.  Tests pin the new ratios (see test_l2v2_oracle.py).

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
# Halfway between LOW_HZ and HIGH_HZ keeps decoding symmetric around the
# input alphabet.  Tunable; not a SPEC-fixed constant.
OUT_SPIKING_THRESHOLD_HZ = 50.0

# ── Mode identifiers ────────────────────────────────────────────────────────
MODE_AND = 0
MODE_NOT = 1
MODE_NAMES = {MODE_AND: "AND", MODE_NOT: "NOT"}

# ── Reward table (rebalanced — see ERRATA v2.1 in module docstring) ────────
#
# Scaled values (× REWARD_SCALE) are what actually arrives at credit:
#     AND spike  = 15.0      AND silent = 0.5
#     NOT spike  = 25.0      NOT silent = 1.0
#
# Anti silent-collapse design: silent rewards (0.5 / 1.0) are intentionally
# below BREATH_PER_WINDOW=1.25 so that "永远 silent" loses ~0.81/window and
# starves out.  Spike rewards (15 / 25) are the only path to net-positive
# credit, forcing evolution to discover the actual logic.
REWARD_SCALE = 0.25
R_AND_SPIKE_RAW = 60.0    # was 20  — 3× to overpower silent attractor on AND(1,1)
R_AND_SILENT_RAW = 2.0    # was  5  — minimal生存奖, sub-breath
R_NOT_SPIKE_RAW = 100.0   # was 50  — high-difficulty premium remains highest
R_NOT_SILENT_RAW = 4.0    # was 10  — minimal生存奖, sub-breath


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
    """Oracle Altar reward — see ERRATA v2.1 in module docstring.

    Spike-correct (target_bit=1) values are amplified vs. SPEC §2.2 to
    overpower the "always silent" attractor that emerges from the truth
    table's class imbalance (3/4 of AND rows + 1/2 of NOT rows expect 0).
    Silent-correct values are dropped below BREATH_PER_WINDOW so that
    永远沉默 → 净亏 → 饿死.
    """
    if mode == MODE_AND:
        return (R_AND_SPIKE_RAW if target_bit == 1 else R_AND_SILENT_RAW) * REWARD_SCALE
    return (R_NOT_SPIKE_RAW if target_bit == 1 else R_NOT_SILENT_RAW) * REWARD_SCALE


def draw_oracle_sample(rng: np.random.Generator) -> OracleSample:
    """Draw one 500 ms stimulus triple and the matching ground truth.

    All four (mode, bit_a, bit_b) combinations are sampled uniformly.  S
    instruction is drawn 50/50 between AND/NOT modes (independent of A,B).
    """
    mode = int(rng.integers(0, 2))            # uniform AND/NOT
    bit_a = int(rng.integers(0, 2))
    bit_b = int(rng.integers(0, 2))
    if mode == MODE_AND:
        target_bit = 1 if (bit_a == 1 and bit_b == 1) else 0
        s_centre = S_AND_HZ
    else:
        # NOT operates on A; B is a distractor (still binary, still drives the network).
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
        reward_wrong=0.0,
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
