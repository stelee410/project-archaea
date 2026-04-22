"""SPEC_L2_V2.0 §2 — Pulsar input + Oracle Altar reward table.

Generates one 500 ms window's *(f_a, f_b, f_s, mode, target_bit, base_reward_table)*.
All agents in the population receive the same stimulus triple per window;
each agent is then judged independently against the truth table.

Mapping from SPEC §2.2:

    | mode | bit_a | bit_b | target_bit | reward(correct) |
    | AND  |   1   |   1   |    1       | +20             |
    | AND  |   1   |   0   |    0       | +5  生存奖励     |
    | AND  |   0   |   1   |    0       | +5              |
    | AND  |   0   |   0   |    0       | +5              |
    | NOT  |   1   |   *   |    0       | +10             |
    | NOT  |   0   |   *   |    1       | +50  高难溢价   |

The raw +20/+50/etc. table would distort the L1 economy
(C_REPRO=200, C_INIT=50, breath=1.25/window; SPEC R_max=5/window).
We multiply by REWARD_SCALE so the *ratios* are preserved while the
absolute magnitudes stay inside the same order as L1 — keeping the
"breath / reproduce in ~30 windows" rhythm that the SPEC §5 economy
was tuned for.

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

# ── Reward table (SPEC §2.2, scaled for L1 economy parity) ─────────────────
#
# To keep "breath = 1.25 / window" and "C_REPRO = 200" tuned to L1, we map:
#   SPEC reward     · 0.25 = effective Credit increment per correct window
# i.e. AND(1,1)=5  · AND(1,0)=1.25 · NOT(1,0)=12.5 · NOT(1,1)=2.5
# which keeps the *ratios* (NOT(1,0) = 10× AND(1,0)) and the absolute
# magnitudes inside [0, R_MAX × small constant].
REWARD_SCALE = 0.25


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
    """SPEC §2.2 Oracle Altar reward (scaled)."""
    if mode == MODE_AND:
        # AND: +20 for both-on, +5 for "silent is correct"
        return (20.0 if target_bit == 1 else 5.0) * REWARD_SCALE
    # NOT: +50 high-difficulty premium for spiking, +10 for silent
    return (50.0 if target_bit == 1 else 10.0) * REWARD_SCALE


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
