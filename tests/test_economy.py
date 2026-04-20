"""Gate B — economic closure (SPEC §6.2)."""

import numpy as np

from archaea.economy import BREATH_PER_WINDOW, C_COST_REPRO, C_INIT, C_REPRO, R_MAX


def simulate_mock_agent(r_per_window: float, n_windows: int) -> np.ndarray:
    credit = C_INIT
    traj = np.zeros(n_windows + 1)
    traj[0] = credit
    defined = True
    for w in range(n_windows):
        r_use = max(0.0, r_per_window) if defined else 0.0
        credit = credit + r_use * R_MAX - BREATH_PER_WINDOW
        traj[w + 1] = credit
    return traj


def test_gate_b_perfect_reproduces():
    r = 1.0
    traj = simulate_mock_agent(r, 120)
    assert traj.max() >= C_REPRO
    crossed = np.where(traj >= C_REPRO)[0]
    assert crossed.size > 0
    t_cross = float(crossed[0]) * 0.5
    assert t_cross <= 35.0, t_cross
    assert traj[-1] > C_REPRO


def test_gate_b_breakeven_band():
    traj = simulate_mock_agent(0.25, 120)
    assert np.all(traj >= 30.0) and np.all(traj <= 80.0), traj


def test_gate_b_deadweight_dies():
    traj = simulate_mock_agent(0.0, 120)
    dead_at = np.where(traj <= 0.0)[0]
    assert dead_at.size > 0
    t_dead = float(dead_at[0]) * 0.5
    assert 15.0 <= t_dead <= 30.0, (t_dead, dead_at[0])


def test_reproduction_deducts_parent_credit():
    c = C_REPRO
    c2 = c - C_COST_REPRO
    assert abs(c2 - 100.0) < 1e-9


# --- Off-SPEC carrying-capacity / shared-budget tests --------------------------


def test_shared_budget_no_haircut_when_under_budget():
    from archaea.economy import shared_budget_rewards

    K = 100  # B = 500 / window
    r_def = np.array([True, True, True, True], dtype=bool)
    r_val = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)  # demand = 4 * 2.5 = 10
    rewards, pressure = shared_budget_rewards(r_def, r_val, K)
    np.testing.assert_allclose(rewards, np.array([2.5, 2.5, 2.5, 2.5]))
    assert pressure == 10.0 / 500.0


def test_shared_budget_proportional_haircut_when_over_budget():
    from archaea.economy import shared_budget_rewards

    K = 2  # B = 10 / window
    r_def = np.array([True, True, True, True], dtype=bool)
    r_val = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    # demand_i = R_MAX * r_i = 5 each, D = 20, ratio = B/D = 0.5
    # → reward_i = 5 * 0.5 = 2.5, sum = 10 = B
    rewards, pressure = shared_budget_rewards(r_def, r_val, K)
    assert pressure == 2.0
    np.testing.assert_allclose(rewards, np.array([2.5, 2.5, 2.5, 2.5]))
    assert abs(rewards.sum() - 5.0 * K) < 1e-12


def test_shared_budget_meritocratic_under_pressure():
    """High-r individuals keep more reward in absolute terms even under haircut."""
    from archaea.economy import shared_budget_rewards

    K = 1  # B = 5 / window
    r_def = np.array([True, True, True], dtype=bool)
    r_val = np.array([1.0, 0.5, 0.1], dtype=np.float64)
    rewards, pressure = shared_budget_rewards(r_def, r_val, K)
    assert pressure > 1.0
    assert rewards[0] > rewards[1] > rewards[2]
    assert abs(rewards.sum() - 5.0) < 1e-12


def test_shared_budget_zero_demand_returns_zero_pressure():
    from archaea.economy import shared_budget_rewards

    r_def = np.zeros(5, dtype=bool)
    r_val = np.zeros(5, dtype=np.float64)
    rewards, pressure = shared_budget_rewards(r_def, r_val, 100)
    assert pressure == 0.0
    np.testing.assert_array_equal(rewards, np.zeros(5))


def test_shared_budget_undefined_skipped():
    from archaea.economy import shared_budget_rewards

    K = 100
    r_def = np.array([True, False, True], dtype=bool)
    r_val = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # only first two * mask matter
    rewards, _ = shared_budget_rewards(r_def, r_val, K)
    assert rewards[1] == 0.0
    assert rewards[0] == 5.0
    assert rewards[2] == 5.0
