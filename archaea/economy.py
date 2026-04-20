"""Credit, breath, reproduction costs (SPEC §4–5)."""

from __future__ import annotations

import numpy as np

C_INIT = 50.0
C_REPRO = 200.0
C_COST_REPRO = 100.0
C_BREATH_PER_STEP = 0.0025
R_MAX = 5.0
WINDOW_MS = 500.0
STEPS_PER_WINDOW = int(WINDOW_MS / 1.0)  # dt=1ms
BREATH_PER_WINDOW = C_BREATH_PER_STEP * STEPS_PER_WINDOW  # 1.25

BUDGET_MODE_NONE = "none"
BUDGET_MODE_SHARED = "shared"
VALID_BUDGET_MODES = (BUDGET_MODE_NONE, BUDGET_MODE_SHARED)


def credit_after_window(credit_old: float, r_agent_defined: bool, r_agent: float) -> float:
    """
    One 500 ms window: + max(0,r)*R_max - breath (SPEC §4.4).
    If fitness undefined, r treated as 0 for reward (SPEC §4.1).
    """
    r_use = max(0.0, r_agent) if r_agent_defined else 0.0
    return credit_old + r_use * R_MAX - BREATH_PER_WINDOW


def shared_budget_rewards(
    r_defined: np.ndarray,
    r_values: np.ndarray,
    carrying_capacity: int,
) -> tuple[np.ndarray, float]:
    """
    Off-SPEC carrying-capacity model (gated by --carrying-capacity / --budget-mode shared).

    Per-window total reward budget:  B = K * R_MAX
    Per-agent demand:                d_i = R_MAX * max(0, r_i)   (0 if undefined)
    Total demand:                    D = sum(d_i)
    If D > B: every agent is rewarded by  d_i * (B/D)  (proportional haircut).
    Otherwise the full demand is paid (equivalent to SPEC).

    Returns (rewards, pressure) where pressure = D / B  (>1 means oversubscribed,
    population is being throttled).  Pressure == 0.0 when nobody has defined
    fitness yet (D == 0).
    """
    K = max(1, int(carrying_capacity))
    B = float(K) * R_MAX
    demand = np.where(r_defined, np.maximum(0.0, r_values), 0.0).astype(np.float64) * R_MAX
    D = float(demand.sum())
    if D <= 0.0:
        return np.zeros_like(demand), 0.0
    pressure = D / B
    ratio = (1.0 / pressure) if pressure > 1.0 else 1.0
    return demand * ratio, pressure


def plain_rewards(
    r_defined: np.ndarray,
    r_values: np.ndarray,
) -> np.ndarray:
    """SPEC §4.4 reward: max(0, r) * R_MAX (0 if undefined). No coupling."""
    return np.where(r_defined, np.maximum(0.0, r_values), 0.0).astype(np.float64) * R_MAX
