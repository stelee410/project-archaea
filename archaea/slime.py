"""
Cyber-slime-mold extension (SPEC v1.1, off-SPEC, opt-in).

Adds three biology-inspired social/cooperative mechanisms on top of the SPEC
v1.0 evolutionary core. All defaults preserve SPEC v1.0 behavior bit-identical
(see ``Population.__init__`` — when ``slime_mold=False`` this module is never
called).

1. **Pheromone field** (cooperation, stigmergy)
   A 2D scalar field over a G×G grid. Each window every fit agent emits at its
   cell, the field then decays and diffuses. Reward in pheromone-rich cells is
   amplified, creating positive feedback for spatial clustering — the classic
   slime-mold trail dynamic.

2. **Horizontal gene transfer (HGT)** (social, lateral learning)
   With small probability each window, a low-credit agent absorbs a fraction of
   a high-credit neighbor's weights. Mimics the lateral DNA exchange that
   defines actual archaea/bacteria. Distinct from reproduction (vertical).

3. **Chemotaxis migration** (social + cooperation)
   Each window agents may step one cell toward the local pheromone gradient,
   with a small random component. Self-organizes spatial structure into
   "foraging networks" that look exactly like Physarum trails.

The field is owned by ``Population``; this module exposes pure functions that
operate on the field + spatial state, plus a ``SlimeConfig`` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Defaults (sensible for pop_max ≈ 200, grid 16×16). Override via SlimeConfig.
DEFAULT_GRID_SIZE = 16
DEFAULT_PHEROMONE_DECAY = 0.05
DEFAULT_PHEROMONE_DIFFUSION = 0.20
DEFAULT_PHEROMONE_EMIT = 0.5
DEFAULT_PHEROMONE_BONUS_K = 0.5
DEFAULT_HGT_PROB = 0.02
DEFAULT_HGT_BLEND = 0.30
DEFAULT_HGT_RADIUS = 1
DEFAULT_HGT_COST = 5.0
DEFAULT_HGT_DONOR_RATIO = 2.0  # donor.credit must be ≥ this × recipient.credit
DEFAULT_MIGRATE_PROB = 0.30


@dataclass
class SlimeConfig:
    """Slime-mold extension parameters (off-SPEC; ``enabled=False`` is the SPEC default)."""

    enabled: bool = False
    grid_size: int = DEFAULT_GRID_SIZE
    pheromone_decay: float = DEFAULT_PHEROMONE_DECAY
    pheromone_diffusion: float = DEFAULT_PHEROMONE_DIFFUSION
    pheromone_emit: float = DEFAULT_PHEROMONE_EMIT
    pheromone_bonus_k: float = DEFAULT_PHEROMONE_BONUS_K
    hgt_enabled: bool = True
    hgt_prob: float = DEFAULT_HGT_PROB
    hgt_blend: float = DEFAULT_HGT_BLEND
    hgt_radius: int = DEFAULT_HGT_RADIUS
    hgt_cost: float = DEFAULT_HGT_COST
    hgt_donor_ratio: float = DEFAULT_HGT_DONOR_RATIO
    migrate_enabled: bool = True
    migrate_prob: float = DEFAULT_MIGRATE_PROB

    def validate(self) -> None:
        if self.grid_size < 4:
            raise ValueError("grid_size must be ≥ 4")
        if not 0.0 <= self.pheromone_decay <= 1.0:
            raise ValueError("pheromone_decay must be in [0, 1]")
        if not 0.0 <= self.pheromone_diffusion <= 1.0:
            raise ValueError("pheromone_diffusion must be in [0, 1]")
        if self.pheromone_emit < 0.0:
            raise ValueError("pheromone_emit must be ≥ 0")
        if self.pheromone_bonus_k < 0.0:
            raise ValueError("pheromone_bonus_k must be ≥ 0")
        if not 0.0 <= self.hgt_prob <= 1.0:
            raise ValueError("hgt_prob must be in [0, 1]")
        if not 0.0 <= self.hgt_blend <= 1.0:
            raise ValueError("hgt_blend must be in [0, 1]")
        if self.hgt_radius < 1:
            raise ValueError("hgt_radius must be ≥ 1")
        if self.hgt_cost < 0.0:
            raise ValueError("hgt_cost must be ≥ 0")
        if self.hgt_donor_ratio < 1.0:
            raise ValueError("hgt_donor_ratio must be ≥ 1")
        if not 0.0 <= self.migrate_prob <= 1.0:
            raise ValueError("migrate_prob must be in [0, 1]")


# --------------------------------------------------------------------- spatial


def random_positions(rng: np.random.Generator, n: int, grid_size: int) -> np.ndarray:
    """Sample n positions uniformly on the G×G torus. Returns shape (n, 2) int32."""
    return rng.integers(0, grid_size, size=(n, 2), dtype=np.int32)


def position_near(
    rng: np.random.Generator, parent_xy: np.ndarray, grid_size: int, max_offset: int = 1
) -> np.ndarray:
    """Place a child within ±max_offset cells of parent (toroidal wrap)."""
    dx = int(rng.integers(-max_offset, max_offset + 1))
    dy = int(rng.integers(-max_offset, max_offset + 1))
    x = (int(parent_xy[0]) + dx) % grid_size
    y = (int(parent_xy[1]) + dy) % grid_size
    return np.array([x, y], dtype=np.int32)


# ------------------------------------------------------------------- pheromone


def new_field(grid_size: int) -> np.ndarray:
    return np.zeros((grid_size, grid_size), dtype=np.float64)


def emit(
    field: np.ndarray,
    positions: np.ndarray,
    fitness: np.ndarray,
    emit_rate: float,
) -> None:
    """In-place: each agent at positions[i] adds emit_rate * max(0, fitness[i])."""
    if positions.size == 0:
        return
    amount = np.maximum(0.0, fitness) * float(emit_rate)
    # Vectorized scatter-add using np.add.at (handles duplicate cells correctly).
    np.add.at(field, (positions[:, 0], positions[:, 1]), amount)


def decay_and_diffuse(field: np.ndarray, decay: float, diffusion: float) -> np.ndarray:
    """
    Apply one-step decay + diffusion to the pheromone field.

    Returns a new field (does not modify input). Uses a 5-point Laplacian on a
    toroidal grid so values can flow in all four cardinal directions; the
    diffusion coefficient is bounded ≤ 1 so the update is unconditionally stable.
    """
    f = field * (1.0 - float(decay))
    if diffusion > 0.0:
        # 4-neighbour Laplacian, periodic boundary
        up = np.roll(f, -1, axis=0)
        down = np.roll(f, 1, axis=0)
        left = np.roll(f, -1, axis=1)
        right = np.roll(f, 1, axis=1)
        lap = 0.25 * (up + down + left + right) - f
        f = f + float(diffusion) * lap
    np.maximum(f, 0.0, out=f)
    return f


def sense(field: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return the local pheromone level at each agent's cell. Shape (n,)."""
    if positions.size == 0:
        return np.zeros(0, dtype=np.float64)
    return field[positions[:, 0], positions[:, 1]].astype(np.float64)


def reward_bonus(local_p: np.ndarray, field_max: float, bonus_k: float) -> np.ndarray:
    """
    Multiplicative reward bonus: 1 + bonus_k * (P_local / P_max).

    Saturates at (1 + bonus_k) on the strongest cell. Returns ones if the field
    is currently empty (no signal to follow).
    """
    if local_p.size == 0 or field_max <= 0.0:
        return np.ones_like(local_p)
    norm = local_p / field_max
    return 1.0 + float(bonus_k) * norm


# ---------------------------------------------------------------- chemotaxis


def chemotaxis_step(
    rng: np.random.Generator,
    field: np.ndarray,
    positions: np.ndarray,
    grid_size: int,
    p_move: float,
) -> np.ndarray:
    """
    For each agent, with prob p_move pick the highest-pheromone neighbour cell
    (incl. self). Ties broken by random. Returns updated positions (n, 2) int32.

    A fully vectorised gradient ascent on the toroidal grid; each agent moves
    at most one cell per call.
    """
    if positions.size == 0 or p_move <= 0.0:
        return positions
    n = positions.shape[0]
    decisions = rng.random(n) < p_move
    if not np.any(decisions):
        return positions

    new_pos = positions.copy()
    # Build 9 neighbour offsets (including self) -> shape (9, 2)
    offsets = np.array(
        [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=np.int32
    )
    # Sample neighbour values for each agent: (n, 9)
    cand_x = (positions[:, 0:1] + offsets[:, 0:1].T) % grid_size  # (n, 9)
    cand_y = (positions[:, 1:2] + offsets[:, 1:2].T) % grid_size  # (n, 9)
    vals = field[cand_x, cand_y]  # (n, 9)
    # Add tiny noise so ties break randomly per call
    vals = vals + rng.uniform(0.0, 1e-9, size=vals.shape)
    best = np.argmax(vals, axis=1)  # (n,)
    chosen_dx = offsets[best, 0]
    chosen_dy = offsets[best, 1]
    moved_x = (positions[:, 0] + chosen_dx) % grid_size
    moved_y = (positions[:, 1] + chosen_dy) % grid_size
    new_pos[decisions, 0] = moved_x[decisions]
    new_pos[decisions, 1] = moved_y[decisions]
    return new_pos.astype(np.int32, copy=False)


# ------------------------------------------------------------------------- HGT


def hgt_pairs(
    rng: np.random.Generator,
    positions: np.ndarray,
    credits: np.ndarray,
    grid_size: int,
    radius: int,
    prob: float,
    donor_ratio: float,
) -> list[tuple[int, int]]:
    """
    Decide HGT (recipient, donor) pairs for this window.

    For each living agent (recipient) with low credit relative to a neighbour
    (donor), emit one pair with probability ``prob``. ``positions`` and
    ``credits`` are *aligned* to one snapshot of living slots (same indexing).

    Returns a list of (recipient_index, donor_index) into the supplied arrays
    — caller maps back to absolute slot ids.
    """
    n = positions.shape[0]
    if n < 2 or prob <= 0.0:
        return []
    pairs: list[tuple[int, int]] = []
    # Pre-roll dice for speed
    rolls = rng.random(n)
    for i in range(n):
        if rolls[i] >= prob:
            continue
        rx, ry = int(positions[i, 0]), int(positions[i, 1])
        # Find candidate donors: |dx|, |dy| ≤ radius (Chebyshev), credit ≥ donor_ratio * mine
        dx = (positions[:, 0] - rx + grid_size // 2) % grid_size - grid_size // 2
        dy = (positions[:, 1] - ry + grid_size // 2) % grid_size - grid_size // 2
        within = (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
        within[i] = False
        if not np.any(within):
            continue
        my_credit = float(credits[i])
        cand_credit = credits.copy()
        cand_credit[~within] = -np.inf
        # Donor must be much richer
        donors_mask = cand_credit >= max(1e-6, donor_ratio * my_credit)
        if not np.any(donors_mask):
            continue
        # Pick the richest donor; tie-break random
        candidates = np.flatnonzero(donors_mask)
        max_c = float(cand_credit[candidates].max())
        top = candidates[np.isclose(cand_credit[candidates], max_c)]
        donor = int(rng.choice(top))
        pairs.append((i, donor))
    return pairs


def blend_weights(
    w_recipient: np.ndarray, w_donor: np.ndarray, blend: float
) -> np.ndarray:
    """Convex blend: (1-η)·self + η·donor. Pure function."""
    eta = float(blend)
    return (1.0 - eta) * w_recipient + eta * w_donor
