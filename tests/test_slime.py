"""Unit tests for the slime-mold extension (SPEC v1.1, off-SPEC)."""

from __future__ import annotations

import numpy as np
import pytest

from archaea.economy import C_INIT
from archaea.population import Population
from archaea.slime import (
    SlimeConfig,
    blend_weights,
    chemotaxis_step,
    decay_and_diffuse,
    emit,
    hgt_pairs,
    new_field,
    sense,
)


# ---------------------------------------------------------------- pheromone


def test_emit_scatter_add_handles_collisions():
    field = new_field(4)
    positions = np.array([[1, 1], [1, 1], [2, 2]], dtype=np.int32)
    fitness = np.array([0.5, 1.0, 0.3])
    emit(field, positions, fitness, emit_rate=2.0)
    # cell (1,1) gets (0.5+1.0)*2.0 = 3.0; (2,2) gets 0.3*2.0 = 0.6
    assert field[1, 1] == pytest.approx(3.0)
    assert field[2, 2] == pytest.approx(0.6)
    assert field[0, 0] == 0.0


def test_emit_clips_negative_fitness():
    field = new_field(4)
    positions = np.array([[0, 0]], dtype=np.int32)
    emit(field, positions, np.array([-0.5]), emit_rate=2.0)
    assert field[0, 0] == 0.0


def test_decay_reduces_total_mass():
    field = np.ones((4, 4), dtype=np.float64)
    after = decay_and_diffuse(field, decay=0.10, diffusion=0.0)
    np.testing.assert_allclose(after, np.full((4, 4), 0.9))


def test_diffusion_conserves_mass_on_torus():
    field = np.zeros((6, 6), dtype=np.float64)
    field[3, 3] = 10.0
    total_before = field.sum()
    after = decay_and_diffuse(field, decay=0.0, diffusion=0.5)
    np.testing.assert_allclose(after.sum(), total_before, rtol=1e-12)
    # Centre shrank, neighbours grew
    assert after[3, 3] < 10.0
    assert after[2, 3] > 0.0
    assert after[4, 3] > 0.0
    assert after[3, 2] > 0.0
    assert after[3, 4] > 0.0


def test_decay_and_diffuse_never_negative():
    rng = np.random.default_rng(0)
    field = rng.uniform(-0.1, 1.0, size=(8, 8))  # arbitrary, may have small negatives
    out = decay_and_diffuse(field, decay=0.05, diffusion=0.30)
    assert (out >= 0.0).all()


def test_sense_returns_local_values():
    field = np.zeros((4, 4))
    field[1, 2] = 5.0
    field[3, 0] = 1.0
    pos = np.array([[1, 2], [3, 0], [0, 0]], dtype=np.int32)
    np.testing.assert_allclose(sense(field, pos), [5.0, 1.0, 0.0])


# ---------------------------------------------------------------- chemotaxis


def test_chemotaxis_climbs_gradient():
    """An agent next to a strong source must step toward it under p_move=1.0."""
    rng = np.random.default_rng(7)
    field = np.zeros((8, 8))
    # Strong source at (3,3); agent at (2,2) (within neighbourhood).
    field[3, 3] = 100.0
    pos = np.array([[2, 2]], dtype=np.int32)
    new_pos = chemotaxis_step(rng, field, pos, grid_size=8, p_move=1.0)
    # Best neighbour cell is exactly (3, 3); the agent should land there.
    np.testing.assert_array_equal(new_pos[0], [3, 3])


def test_chemotaxis_respects_p_move_zero():
    rng = np.random.default_rng(0)
    field = np.zeros((8, 8))
    field[1, 1] = 1.0
    pos = np.array([[5, 5]], dtype=np.int32)
    new_pos = chemotaxis_step(rng, field, pos, grid_size=8, p_move=0.0)
    np.testing.assert_array_equal(new_pos, pos)


# ----------------------------------------------------------------------- HGT


def test_hgt_pairs_picks_richer_neighbour():
    """Recipient at (0,0), donor at (1,0) with 10× credit."""
    rng = np.random.default_rng(123)
    pos = np.array([[0, 0], [1, 0]], dtype=np.int32)
    cred = np.array([10.0, 100.0])
    pairs = hgt_pairs(
        rng, pos, cred, grid_size=8, radius=1, prob=1.0, donor_ratio=2.0
    )
    assert len(pairs) >= 1
    # Recipient is the poor one, donor is the rich one
    poor_donor_pairs = [(r, d) for (r, d) in pairs if r == 0 and d == 1]
    assert len(poor_donor_pairs) >= 1


def test_hgt_pairs_skips_when_no_richer_neighbour():
    """Equal credit → nobody is ≥ 2× anyone else → no transfers."""
    rng = np.random.default_rng(123)
    pos = np.array([[0, 0], [1, 0]], dtype=np.int32)
    cred = np.array([100.0, 100.0])
    pairs = hgt_pairs(
        rng, pos, cred, grid_size=8, radius=1, prob=1.0, donor_ratio=2.0
    )
    assert pairs == []


def test_hgt_pairs_radius_filter():
    """Donor outside radius is invisible."""
    rng = np.random.default_rng(123)
    pos = np.array([[0, 0], [4, 4]], dtype=np.int32)  # Far apart
    cred = np.array([10.0, 1000.0])
    pairs = hgt_pairs(
        rng, pos, cred, grid_size=16, radius=1, prob=1.0, donor_ratio=2.0
    )
    assert pairs == []


def test_blend_weights_convex_combination():
    a = np.zeros(220)
    b = np.ones(220)
    out = blend_weights(a, b, 0.30)
    np.testing.assert_allclose(out, np.full(220, 0.30))


# -------------------------------------------------------------- integration


def test_population_slime_disabled_matches_default_behaviour():
    """slime=disabled must keep SPEC v1.0 behaviour (no extra fields side-effects)."""
    rng_a = np.random.default_rng(42)
    pop_a = Population(20, rng_a, n_initial=20)

    rng_b = np.random.default_rng(42)
    pop_b = Population(20, rng_b, n_initial=20, slime=SlimeConfig(enabled=False))

    info_a = pop_a.step_window()
    info_b = pop_b.step_window()
    # Same RNG → same f_in draw, same dynamics
    assert info_a["f_in"] == info_b["f_in"]
    assert info_a["births"] == info_b["births"]
    assert info_a["deaths"] == info_b["deaths"]
    assert info_a["r_max"] == info_b["r_max"]
    np.testing.assert_allclose(pop_a.credit, pop_b.credit)


def test_population_slime_enabled_runs_and_emits_pheromone():
    rng = np.random.default_rng(0)
    cfg = SlimeConfig(
        enabled=True,
        grid_size=8,
        pheromone_decay=0.05,
        pheromone_diffusion=0.20,
        pheromone_emit=2.0,
        pheromone_bonus_k=0.5,
        hgt_enabled=True,
        hgt_prob=0.10,
        migrate_enabled=True,
        migrate_prob=0.50,
    )
    pop = Population(30, rng, n_initial=30, slime=cfg)
    # Force-stamp fitness history so first window can emit (without waiting 40 windows).
    pop._hc[:30] = 40
    for s in range(30):
        pop._fin[s] = np.linspace(10.0, 100.0, 40)
        pop._fout[s] = np.linspace(5.0, 50.0, 40)  # perfect linear → r=1
    info = pop.step_window()
    assert info["pheromone_max"] >= 0.0
    # After at least one window, total pheromone mass must be > 0 since fitness was forced positive.
    assert pop.pheromone.sum() > 0.0
    # Positions stored as int32 grid coords
    assert pop.positions.dtype == np.int32
    assert (pop.positions[:30] >= 0).all()
    assert (pop.positions[:30] < cfg.grid_size).all()


def test_population_slime_validates_bad_config():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        Population(20, rng, slime=SlimeConfig(enabled=True, grid_size=2))
    with pytest.raises(ValueError):
        Population(20, rng, slime=SlimeConfig(enabled=True, pheromone_decay=2.0))
    with pytest.raises(ValueError):
        Population(20, rng, slime=SlimeConfig(enabled=True, hgt_donor_ratio=0.5))
