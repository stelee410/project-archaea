"""Evolution task identifiers and routing helpers.

L1  : SPEC §3.1 single-channel rate tracking. Bit-identical to original SPEC.
L2v2: SPEC_L2_V2.0 logic gating with three-channel input (A=4, B=4, S=2)
      and instruction-conditioned reward table (oracle).
"""

from __future__ import annotations

TASK_L1 = "l1"
TASK_L2V2 = "l2v2_ctrl"

DEFAULT_TASK = TASK_L1
VALID_TASKS = (TASK_L1, TASK_L2V2)

# Channel split for L2v2.  10 inputs are partitioned into:
#   A: 4 input neurons (data 1)
#   B: 4 input neurons (data 2)
#   S: 2 input neurons (selector / instruction)
# Sum must equal N_INPUT (=10).  If you change this, also change
# stimulus.poisson_three_channels and oracle.SLOT_INDEX.
N_CH_A = 4
N_CH_B = 4
N_CH_S = 2


def validate_task(task: str) -> str:
    t = str(task)
    if t not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}")
    return t


def is_logic_task(task: str) -> bool:
    return validate_task(task) == TASK_L2V2
