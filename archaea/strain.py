"""
SPEC_L2_V3.0 — 「菌株（Strain）」存储模块。

一个菌株 = 某个培养皿在某个时刻的「全部活体快照」（或 top-K 子集）。
是 admixture experiment（杂交皿）的基本物料：用户先在两个独立环境里
养出 AND-学家 / NOT-学家两个菌株，再倒在一起，让 HGT 在杂交期里
把基因混合，自然涌现「同时会 AND 和 NOT」的双修个体。

与 ``archaea.champions`` 的区别：
- champions 服务的是「冷启动推理」(L1 rate tracking) — 只存 top-K 精英；
- strain 服务的是「下一次仿真的 founders」— 存全部活体（最多 pop_max），
  且记录 task / source colony / accuracy 等 admixture 决策需要的元数据。

存储格式：``checkpoints/strains/<id>.npz`` 单文件 + 同名 ``<id>.json``
缓存元数据用于快速列表浏览（避免加载 weights 才能看 name/acc）。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .neuron import N_WEIGHTS
from .task import TASK_L2V2, validate_task

SPEC_VERSION = "L2.V3.0"

# Default storage root.  Tests override via constructor argument.
DEFAULT_STRAIN_DIR = Path(__file__).resolve().parents[1] / "checkpoints" / "strains"


@dataclass
class StrainMeta:
    """Lightweight metadata — what UI lists need without loading weights."""

    id: str
    name: str
    task: str
    n_agents: int
    t_sim: float
    source_seed: int
    source_difficulty: str | None  # only meaningful for L2v2
    acc_and_pop_at_save: float | None
    acc_not_pop_at_save: float | None
    fitness_mean: float
    fitness_max: float
    note: str
    created_at: str
    spec_version: str = SPEC_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrainMeta":
        return cls(
            id=str(d["id"]),
            name=str(d["name"]),
            task=validate_task(d["task"]),
            n_agents=int(d["n_agents"]),
            t_sim=float(d["t_sim"]),
            source_seed=int(d["source_seed"]),
            source_difficulty=(None if d.get("source_difficulty") is None else str(d["source_difficulty"])),
            acc_and_pop_at_save=(None if d.get("acc_and_pop_at_save") is None else float(d["acc_and_pop_at_save"])),
            acc_not_pop_at_save=(None if d.get("acc_not_pop_at_save") is None else float(d["acc_not_pop_at_save"])),
            fitness_mean=float(d.get("fitness_mean", 0.0)),
            fitness_max=float(d.get("fitness_max", 0.0)),
            note=str(d.get("note", "")),
            created_at=str(d["created_at"]),
            spec_version=str(d.get("spec_version", SPEC_VERSION)),
        )


@dataclass
class Strain:
    """Full payload — meta + weights + per-agent fitness."""

    meta: StrainMeta
    weights: np.ndarray  # (K, 220) float64
    fitness_at_save: np.ndarray  # (K,) float64 — NaN if undefined

    def __post_init__(self) -> None:
        if self.weights.ndim != 2 or self.weights.shape[1] != N_WEIGHTS:
            raise ValueError(
                f"weights must be (K, {N_WEIGHTS}), got {self.weights.shape}"
            )
        if self.fitness_at_save.shape != (self.weights.shape[0],):
            raise ValueError(
                f"fitness_at_save must be (K,) where K={self.weights.shape[0]}, "
                f"got {self.fitness_at_save.shape}"
            )
        if int(self.meta.n_agents) != int(self.weights.shape[0]):
            raise ValueError(
                f"meta.n_agents={self.meta.n_agents} != weights K={self.weights.shape[0]}"
            )


# ── Save / Load ────────────────────────────────────────────────────────────


def _new_strain_id() -> str:
    """Short URL-friendly id (12 hex chars from a UUID4 — collision-safe enough for one user)."""
    return uuid.uuid4().hex[:12]


def save_strain_from_population(
    pop,
    *,
    name: str,
    note: str = "",
    t_sim: float = 0.0,
    source_seed: int = -1,
    source_difficulty: str | None = None,
    acc_and_pop: float | None = None,
    acc_not_pop: float | None = None,
    storage_dir: str | Path | None = None,
    strain_id: str | None = None,
) -> StrainMeta:
    """Snapshot all living agents in ``pop`` into a new strain.

    Returns the saved metadata.  Raises if no living agents.

    Note we save *all* living agents (not top-K like champions.py) — admixture
    typically wants the whole gene pool of the source culture so HGT can mix
    a representative sample, not just the elites.
    """
    living = pop.living_indices()
    if living.size == 0:
        raise RuntimeError("no living agents to snapshot")

    weights = np.ascontiguousarray(pop.weights[living], dtype=np.float64)
    fitness = np.array(
        [
            pop._fitness_slot(int(s)) if pop._fitness_defined(int(s)) else float("nan")
            for s in living.tolist()
        ],
        dtype=np.float64,
    )
    finite = fitness[np.isfinite(fitness)]
    fmean = float(finite.mean()) if finite.size else 0.0
    fmax = float(finite.max()) if finite.size else 0.0

    sid = strain_id or _new_strain_id()
    meta = StrainMeta(
        id=sid,
        name=str(name).strip() or f"strain-{sid}",
        task=validate_task(pop.task),
        n_agents=int(weights.shape[0]),
        t_sim=float(t_sim),
        source_seed=int(source_seed),
        source_difficulty=(str(source_difficulty) if source_difficulty else None),
        acc_and_pop_at_save=(None if acc_and_pop is None else float(acc_and_pop)),
        acc_not_pop_at_save=(None if acc_not_pop is None else float(acc_not_pop)),
        fitness_mean=fmean,
        fitness_max=fmax,
        note=str(note),
        created_at=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    )

    root = Path(storage_dir) if storage_dir is not None else DEFAULT_STRAIN_DIR
    root.mkdir(parents=True, exist_ok=True)

    npz_path = root / f"{sid}.npz"
    json_path = root / f"{sid}.json"
    np.savez_compressed(
        npz_path,
        weights=weights,
        fitness_at_save=fitness,
        # store meta inline too — single-file rehydration even if .json is lost
        _meta_json=np.array(json.dumps(meta.to_dict()), dtype=object),
    )
    json_path.write_text(json.dumps(meta.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def load_strain(strain_id: str, storage_dir: str | Path | None = None) -> Strain:
    """Load a strain by id (full payload, including weights)."""
    root = Path(storage_dir) if storage_dir is not None else DEFAULT_STRAIN_DIR
    npz_path = root / f"{strain_id}.npz"
    json_path = root / f"{strain_id}.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"strain not found: {strain_id} (no {npz_path})")

    # Prefer the sidecar JSON for meta (cheaper, hand-editable for note fixes);
    # fall back to the in-npz copy if missing.
    if json_path.exists():
        meta = StrainMeta.from_dict(json.loads(json_path.read_text(encoding="utf-8")))
        d = np.load(npz_path, allow_pickle=False)
    else:
        d = np.load(npz_path, allow_pickle=True)
        if "_meta_json" not in d.files:
            raise RuntimeError(
                f"strain {strain_id} has neither sidecar JSON nor inline meta"
            )
        meta = StrainMeta.from_dict(json.loads(str(d["_meta_json"])))

    weights = np.asarray(d["weights"], dtype=np.float64)
    fitness = np.asarray(d["fitness_at_save"], dtype=np.float64)
    return Strain(meta=meta, weights=weights, fitness_at_save=fitness)


def list_strains(storage_dir: str | Path | None = None) -> list[StrainMeta]:
    """Return all strain metadata sorted by created_at (newest first).

    Reads only the .json sidecars — never touches the .npz weights, so this is
    cheap even with many strains."""
    root = Path(storage_dir) if storage_dir is not None else DEFAULT_STRAIN_DIR
    if not root.exists():
        return []
    out: list[StrainMeta] = []
    for p in root.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append(StrainMeta.from_dict(d))
        except (json.JSONDecodeError, KeyError, ValueError):
            # Skip corrupt sidecars rather than blowing up the whole list.
            continue
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


def delete_strain(strain_id: str, storage_dir: str | Path | None = None) -> bool:
    """Remove both .npz and .json for the given strain.  Returns True if anything removed."""
    root = Path(storage_dir) if storage_dir is not None else DEFAULT_STRAIN_DIR
    removed = False
    for ext in (".npz", ".json"):
        p = root / f"{strain_id}{ext}"
        if p.exists():
            os.remove(p)
            removed = True
    return removed
