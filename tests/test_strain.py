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


# ── 4. SPEC_L2_V3.4 — 3-phase admixture protocol ───────────────────────


def _make_admixture_pop(
    *,
    commensal_s: float,
    exchange_s: float,
    phase2_blend: float = 0.05,
    phase2_prob_mul: float = 1.0,
    base_hgt_prob: float = 0.05,
    base_hgt_blend: float = 0.30,
    pop_max: int = 80,
    seed: int = 123,
) -> Population:
    rng = np.random.default_rng(seed)
    slime = SlimeConfig(
        enabled=True,
        grid_size=8,
        hgt_enabled=True,
        hgt_prob=base_hgt_prob,
        hgt_blend=base_hgt_blend,
        hgt_radius=3,
        hgt_cost=0.0,
        hgt_donor_ratio=1.0,
        pheromone_bonus_k=0.0,
        migrate_enabled=False,
    )
    return Population(
        pop_max=pop_max,
        rng=rng,
        n_initial=pop_max,
        task=TASK_L2V2,
        slime=slime,
        admixture_commensal_s=commensal_s,
        admixture_exchange_s=exchange_s,
        admixture_phase2_blend=phase2_blend,
        admixture_phase2_prob_mul=phase2_prob_mul,
    )


def test_admixture_commensal_phase_disables_hgt() -> None:
    """Phase 1 must completely suppress HGT — no transfers, no eff_hgt_prob."""
    pop = _make_admixture_pop(commensal_s=5.0, exchange_s=5.0)
    counts = []
    for _ in range(10):  # 5.0 s = strictly inside commensal phase
        info = pop.step_window()
        assert info["admixture_phase"] == 1, info["admixture_phase"]
        assert info["admixture_active"] is True
        assert info["eff_hgt_prob"] == 0.0
        assert info["eff_hgt_blend"] == 0.0
        counts.append(int(info["hgt_count"]))
    assert sum(counts) == 0, (
        f"commensal phase 1 must disable HGT entirely, got {sum(counts)} transfers"
    )


def test_admixture_exchange_phase_uses_low_blend() -> None:
    """Phase 2 reports the configured low blend, not the slime baseline."""
    pop = _make_admixture_pop(
        commensal_s=2.0, exchange_s=4.0, phase2_blend=0.07, phase2_prob_mul=2.0,
        base_hgt_blend=0.30, base_hgt_prob=0.05,
    )
    # Skip past phase 1.
    for _ in range(4):  # 2.0 s
        pop.step_window()
    info = pop.step_window()
    assert info["admixture_phase"] == 2
    assert info["admixture_active"] is True
    assert info["eff_hgt_blend"] == pytest.approx(0.07)
    assert info["eff_hgt_prob"] == pytest.approx(0.05 * 2.0)


def test_admixture_restored_phase_returns_to_baseline() -> None:
    """Phase 3 (after exchange ends) restores baseline blend / prob."""
    pop = _make_admixture_pop(
        commensal_s=1.0, exchange_s=1.0, phase2_blend=0.05, phase2_prob_mul=4.0,
        base_hgt_blend=0.30, base_hgt_prob=0.05,
    )
    # Run all the way past commensal_s + exchange_s = 2.0 s.
    for _ in range(8):  # 4.0 s
        pop.step_window()
    info = pop.step_window()
    assert info["admixture_phase"] == 3
    assert info["admixture_active"] is False
    assert info["eff_hgt_blend"] == pytest.approx(0.30)
    assert info["eff_hgt_prob"] == pytest.approx(0.05)


def test_admixture_disabled_protocol_runs_baseline_from_t0() -> None:
    """commensal_s=0 and exchange_s=0 must mean phase=3 from the very first window."""
    pop = _make_admixture_pop(
        commensal_s=0.0, exchange_s=0.0, base_hgt_blend=0.30, base_hgt_prob=0.05,
    )
    info = pop.step_window()
    assert info["admixture_phase"] == 3
    assert info["admixture_active"] is False
    assert info["eff_hgt_blend"] == pytest.approx(0.30)
    assert info["eff_hgt_prob"] == pytest.approx(0.05)


def test_admixture_phase_transitions_are_monotonic() -> None:
    """Walk through 1 → 2 → 3 across the boundaries; never goes backwards."""
    pop = _make_admixture_pop(commensal_s=1.0, exchange_s=2.0)
    seen: list[int] = []
    for _ in range(10):  # 5.0 s — covers all three phases
        info = pop.step_window()
        seen.append(int(info["admixture_phase"]))
    # Must contain at least one of each phase, in non-decreasing order.
    assert 1 in seen and 2 in seen and 3 in seen, seen
    assert seen == sorted(seen), f"phase regressed: {seen}"


def test_admixture_exchange_blend_is_applied_to_recipients() -> None:
    """During phase 2, blend_weights uses phase2_blend (not slime.hgt_blend).

    We verify by setting phase2_blend=0 (no change) and confirming recipient
    weights remain identical to themselves after a window where HGT events
    fire — proves the eff_blend is wired into the actual blend_weights call.
    """
    rng = np.random.default_rng(0)
    slime = SlimeConfig(
        enabled=True,
        grid_size=8,
        hgt_enabled=True,
        hgt_prob=1.0,           # force HGT every neighbor pair
        hgt_blend=0.30,         # baseline that we must NOT see during phase 2
        hgt_radius=8,
        hgt_cost=0.0,
        hgt_donor_ratio=1.0,
        pheromone_bonus_k=0.0,
        migrate_enabled=False,
    )
    pop = Population(
        pop_max=20,
        rng=rng,
        n_initial=20,
        task=TASK_L2V2,
        slime=slime,
        admixture_commensal_s=0.0,    # skip phase 1
        admixture_exchange_s=10.0,    # stay in phase 2
        admixture_phase2_blend=0.0,   # no blending should occur
        admixture_phase2_prob_mul=1.0,
    )
    snap = pop.weights[pop.alive].copy()
    info = pop.step_window()
    assert info["admixture_phase"] == 2
    # Even with hgt_count > 0, weights cannot have changed when blend == 0.
    assert np.allclose(pop.weights[pop.alive], snap), (
        "phase2_blend=0 must leave recipient weights untouched"
    )
