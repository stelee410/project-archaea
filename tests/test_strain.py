"""SPEC_L2_V3.0 — strain (菌株) save/load + admixture experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from archaea.neuron import N_WEIGHTS
from archaea.population import FounderInjection, Population
from archaea.slime import SlimeConfig
from archaea.strain import (
    SPEC_VERSION,
    Strain,
    StrainMeta,
    delete_strain,
    list_strains,
    load_strain,
    save_strain_from_population,
)
from archaea.task import TASK_L1, TASK_L2V2


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_l2v2_pop(n: int = 30, seed: int = 7) -> Population:
    rng = np.random.default_rng(seed)
    return Population(
        pop_max=n, rng=rng, n_initial=n, task=TASK_L2V2, task_difficulty="balanced"
    )


def _make_l1_pop(n: int = 20, seed: int = 11) -> Population:
    rng = np.random.default_rng(seed)
    return Population(pop_max=n, rng=rng, n_initial=n, task=TASK_L1)


# ── 1. Save / load roundtrip ────────────────────────────────────────────


def test_strain_save_load_roundtrip(tmp_path: Path) -> None:
    """save_strain → load_strain returns byte-identical weights + same meta."""
    pop = _make_l2v2_pop(n=25)
    meta = save_strain_from_population(
        pop,
        name="test-strain",
        note="hello",
        t_sim=12.5,
        source_seed=42,
        source_difficulty="balanced",
        acc_and_pop=0.42,
        acc_not_pop=0.31,
        storage_dir=tmp_path,
    )
    assert meta.n_agents == 25
    assert meta.task == TASK_L2V2
    assert meta.spec_version == SPEC_VERSION

    loaded = load_strain(meta.id, storage_dir=tmp_path)
    assert loaded.weights.shape == (25, N_WEIGHTS)
    np.testing.assert_array_equal(loaded.weights, pop.weights[pop.living_indices()])
    assert loaded.meta.name == "test-strain"
    assert loaded.meta.note == "hello"
    assert loaded.meta.t_sim == 12.5
    assert loaded.meta.source_seed == 42
    assert loaded.meta.source_difficulty == "balanced"
    assert loaded.meta.acc_and_pop_at_save == pytest.approx(0.42)
    assert loaded.meta.acc_not_pop_at_save == pytest.approx(0.31)


def test_strain_meta_json_consistent(tmp_path: Path) -> None:
    """The .json sidecar matches what's serialized inside the .npz."""
    pop = _make_l2v2_pop(n=10)
    meta = save_strain_from_population(pop, name="m", storage_dir=tmp_path)
    sidecar = json.loads((tmp_path / f"{meta.id}.json").read_text(encoding="utf-8"))
    assert sidecar["id"] == meta.id
    assert sidecar["n_agents"] == 10
    assert sidecar["task"] == TASK_L2V2


def test_strain_save_no_living_raises(tmp_path: Path) -> None:
    pop = _make_l2v2_pop(n=5)
    pop.alive[:] = False  # kill all
    with pytest.raises(RuntimeError, match="no living"):
        save_strain_from_population(pop, name="x", storage_dir=tmp_path)


def test_strain_list_sorted_newest_first(tmp_path: Path) -> None:
    pop = _make_l2v2_pop(n=5)
    a = save_strain_from_population(pop, name="alpha", storage_dir=tmp_path)
    # Tweak created_at for deterministic ordering
    j = json.loads((tmp_path / f"{a.id}.json").read_text(encoding="utf-8"))
    j["created_at"] = "2020-01-01T00:00:00+00:00"
    (tmp_path / f"{a.id}.json").write_text(json.dumps(j), encoding="utf-8")

    b = save_strain_from_population(pop, name="beta", storage_dir=tmp_path)
    j = json.loads((tmp_path / f"{b.id}.json").read_text(encoding="utf-8"))
    j["created_at"] = "2026-01-01T00:00:00+00:00"
    (tmp_path / f"{b.id}.json").write_text(json.dumps(j), encoding="utf-8")

    lst = list_strains(storage_dir=tmp_path)
    assert [m.name for m in lst] == ["beta", "alpha"]


def test_strain_delete_removes_both_files(tmp_path: Path) -> None:
    pop = _make_l2v2_pop(n=5)
    meta = save_strain_from_population(pop, name="x", storage_dir=tmp_path)
    assert (tmp_path / f"{meta.id}.npz").exists()
    assert (tmp_path / f"{meta.id}.json").exists()

    assert delete_strain(meta.id, storage_dir=tmp_path) is True
    assert not (tmp_path / f"{meta.id}.npz").exists()
    assert not (tmp_path / f"{meta.id}.json").exists()
    # Idempotent
    assert delete_strain(meta.id, storage_dir=tmp_path) is False


# ── 2. spawn_from_strains via founders= ─────────────────────────────────


def test_founders_two_strains_50_50_fraction(tmp_path: Path) -> None:
    """50% A + 50% B → 50 slots from A pool, 50 slots from B pool."""
    pop_a = _make_l2v2_pop(n=10, seed=1)
    pop_b = _make_l2v2_pop(n=10, seed=2)
    # Make pools deterministic constants so we can identify provenance.
    pop_a.weights[pop_a.living_indices()] = 1.0
    pop_b.weights[pop_b.living_indices()] = -1.0
    a_w = pop_a.weights[pop_a.living_indices()]
    b_w = pop_b.weights[pop_b.living_indices()]

    rng = np.random.default_rng(99)
    pop = Population(
        pop_max=200,
        rng=rng,
        n_initial=100,
        task=TASK_L2V2,
        founders=[
            FounderInjection(weights=a_w, fraction=0.5, label="A"),
            FounderInjection(weights=b_w, fraction=0.5, label="B"),
        ],
    )
    living = pop.living_indices()
    assert living.size == 100
    n_a = int((pop.weights[living][:, 0] == 1.0).sum())
    n_b = int((pop.weights[living][:, 0] == -1.0).sum())
    assert n_a == 50
    assert n_b == 50


def test_founders_fraction_sum_lt_1_remainder_random(tmp_path: Path) -> None:
    """0.3 + 0.4 = 0.7 → 30 from A + 40 from B + 30 from random init."""
    rng_pool = np.random.default_rng(0)
    a_w = np.full((5, N_WEIGHTS), 1.0)
    b_w = np.full((5, N_WEIGHTS), -1.0)

    pop = Population(
        pop_max=100,
        rng=np.random.default_rng(7),
        n_initial=100,
        task=TASK_L2V2,
        founders=[
            FounderInjection(weights=a_w, fraction=0.3),
            FounderInjection(weights=b_w, fraction=0.4),
        ],
    )
    w0 = pop.weights[pop.living_indices()][:, 0]
    n_a = int((w0 == 1.0).sum())
    n_b = int((w0 == -1.0).sum())
    n_other = 100 - n_a - n_b
    assert n_a == 30
    assert n_b == 40
    # Remainder is random uniform in [-0.5, 1.5] for L2v2 — neither 1.0 nor -1.0
    # almost surely (probability of exact match is zero in continuous distribution).
    assert n_other == 30


def test_founders_fraction_sum_gt_1_raises() -> None:
    a_w = np.zeros((3, N_WEIGHTS))
    b_w = np.zeros((3, N_WEIGHTS))
    with pytest.raises(ValueError, match="sum to"):
        Population(
            pop_max=10,
            rng=np.random.default_rng(0),
            n_initial=10,
            task=TASK_L2V2,
            founders=[
                FounderInjection(weights=a_w, fraction=0.6),
                FounderInjection(weights=b_w, fraction=0.6),
            ],
        )


def test_founders_invalid_shape_raises() -> None:
    bad_w = np.zeros((4, N_WEIGHTS - 1))
    with pytest.raises(ValueError, match="must be"):
        Population(
            pop_max=5,
            rng=np.random.default_rng(0),
            n_initial=5,
            task=TASK_L2V2,
            founders=[FounderInjection(weights=bad_w, fraction=1.0)],
        )


def test_strain_save_then_spawn_preserves_population(tmp_path: Path) -> None:
    """End-to-end: save a pop, load, spawn into a fresh pop, weights survive."""
    pop = _make_l2v2_pop(n=15)
    meta = save_strain_from_population(pop, name="src", storage_dir=tmp_path)
    strain = load_strain(meta.id, storage_dir=tmp_path)

    rng = np.random.default_rng(0)
    new_pop = Population(
        pop_max=30,
        rng=rng,
        n_initial=15,
        task=TASK_L2V2,
        founders=[FounderInjection(weights=strain.weights, fraction=1.0)],
    )
    living = new_pop.living_indices()
    assert living.size == 15
    # Each slot should equal *some* row of the source pool — exact rows depend
    # on the with-replacement sampling, but the value set must be a subset.
    src_set = {tuple(row.round(8)) for row in strain.weights}
    for w in new_pop.weights[living]:
        assert tuple(w.round(8)) in src_set


# ── 3. Cross-task incompatibility (caught at runtime layer; here we just
#       confirm the meta carries task so the runtime check has data to work on)


def test_strain_carries_task_for_runtime_check(tmp_path: Path) -> None:
    pop_l1 = _make_l1_pop(n=8)
    meta = save_strain_from_population(pop_l1, name="l1-strain", storage_dir=tmp_path)
    assert meta.task == TASK_L1


# ── 4. Admixture window boosts HGT count ───────────────────────────────


def test_admixture_window_boosts_hgt() -> None:
    """In the admixture window, eff_hgt_prob = base × multiplier (capped at 1)."""
    rng = np.random.default_rng(123)
    slime = SlimeConfig(
        enabled=True,
        grid_size=8,
        hgt_enabled=True,
        hgt_prob=0.05,
        hgt_blend=0.3,
        hgt_radius=3,
        hgt_cost=0.0,        # avoid starving recipients during the test
        hgt_donor_ratio=1.0, # SlimeConfig minimum; any donor with ≥ recipient credit counts
        pheromone_bonus_k=0.0,
        migrate_enabled=False,
    )
    pop = Population(
        pop_max=80,
        rng=rng,
        n_initial=80,
        task=TASK_L2V2,
        slime=slime,
        admixture_window_s=10.0,    # first 10 sim-seconds = boosted HGT
        admixture_hgt_multiplier=10.0,
    )
    # Run 5 windows inside the window, then 5 outside it (each window = 0.5 s).
    inside_counts = []
    for _ in range(20):  # 10 seconds
        info = pop.step_window()
        inside_counts.append(int(info["hgt_count"]))
        # admixture_active should be True for all windows where the *previous*
        # tick still had t < window; the flag flips at the boundary.
    # Window has ended now (t_sim = 10.0).
    assert pop._t_sim_seconds == pytest.approx(10.0)
    outside_counts = []
    for _ in range(20):  # next 10 seconds
        info = pop.step_window()
        outside_counts.append(int(info["hgt_count"]))
        assert info["admixture_active"] is False
    inside_total = sum(inside_counts)
    outside_total = sum(outside_counts)
    # We don't assert exact ratios — Poisson noise + drift in pop credit makes
    # it noisy — but the boosted regime must produce strictly more HGT events
    # than the unboosted, otherwise the multiplier is silently broken.
    assert inside_total > outside_total, (
        f"admixture window failed to boost HGT: "
        f"inside={inside_total}, outside={outside_total}"
    )


def test_admixture_window_zero_means_no_boost() -> None:
    """admixture_window_s=0 → eff_hgt_prob == base every window."""
    pop = Population(
        pop_max=20,
        rng=np.random.default_rng(0),
        n_initial=20,
        task=TASK_L2V2,
        slime=SlimeConfig(enabled=True, hgt_prob=0.05, pheromone_bonus_k=0.0),
        admixture_window_s=0.0,
        admixture_hgt_multiplier=10.0,
    )
    info = pop.step_window()
    assert info["admixture_active"] is False
    assert info["eff_hgt_prob"] == pytest.approx(0.05)
