"""
Save the population's best individuals as a portable «DNA» archive,
load it later for inference (rate-tracking service).

File format (.npz, single file per archive):
- weights:        float64, shape (K, 220)
- fitness:        float64, shape (K,)        — Pearson r at dump time (NaN if undefined)
- credit:         float64, shape (K,)        — Credit at dump time
- source_slot:    int32,   shape (K,)        — original slot id in the population
- t_sim:          float64, scalar            — sim seconds at dump time
- seed:           int64,   scalar            — RNG seed of the originating run (-1 if unknown)
- pop_max:        int64,   scalar            — population cap of the originating run
- spec_version:   "L1.0"                     — bumped if SPEC §2 constants change
- created_at:     ISO 8601 UTC string

Inference convention (matches SPEC §3 / §1.3):
- 10 input neurons, independent Poisson trains at the **same** rate f_in (Hz).
- 1-output rate over a window = (output spikes) / (window seconds).
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable

import numpy as np

from .neuron import (
    DT_MS,
    N_INPUT,
    N_WEIGHTS,
    NetworkBatch,
)
from .stimulus import poisson_spikes_window

SPEC_VERSION = "L1.0"


def _slot_fitness(pop, slot: int) -> float:
    if not pop._fitness_defined(int(slot)):
        return float("nan")
    return float(pop._fitness_slot(int(slot)))


def save_champions(
    pop,
    path: str,
    top_k: int = 10,
    *,
    t_sim: float = 0.0,
    seed: int = -1,
) -> str:
    """Pick top-K living agents by fitness (NaN-fitness pushed to the end), save to ``path``."""
    living = pop.living_indices()
    if living.size == 0:
        raise RuntimeError("no living agents to dump")
    scores = np.array(
        [_slot_fitness(pop, int(s)) for s in living.tolist()],
        dtype=np.float64,
    )
    keys = np.where(np.isnan(scores), -np.inf, scores)
    order = np.argsort(keys, kind="stable")[::-1]
    pick = order[: min(int(top_k), order.size)]
    chosen = living[pick]

    weights = np.ascontiguousarray(pop.weights[chosen], dtype=np.float64)
    fitness = scores[pick].astype(np.float64)
    credit = pop.credit[chosen].astype(np.float64)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=weights,
        fitness=fitness,
        credit=credit,
        source_slot=chosen.astype(np.int32),
        t_sim=np.float64(t_sim),
        seed=np.int64(seed),
        pop_max=np.int64(pop.pop_max),
        spec_version=np.array(SPEC_VERSION),
        created_at=np.array(_dt.datetime.now(tz=_dt.timezone.utc).isoformat()),
    )
    return str(path)


class ChampionEnsemble:
    """Loaded champions; vectorized inference for K agents at once."""

    __slots__ = (
        "weights",
        "fitness",
        "credit",
        "source_slot",
        "t_sim",
        "seed",
        "pop_max",
        "spec_version",
        "created_at",
        "_net",
    )

    def __init__(
        self,
        weights: np.ndarray,
        fitness: np.ndarray | None = None,
        credit: np.ndarray | None = None,
        source_slot: np.ndarray | None = None,
        t_sim: float = 0.0,
        seed: int = -1,
        pop_max: int = 0,
        spec_version: str = SPEC_VERSION,
        created_at: str = "",
    ):
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] != N_WEIGHTS:
            raise ValueError(f"weights must be (K, {N_WEIGHTS}), got {w.shape}")
        self.weights = w
        k = w.shape[0]
        self.fitness = (
            np.full(k, np.nan, dtype=np.float64) if fitness is None else np.asarray(fitness, dtype=np.float64)
        )
        self.credit = (
            np.zeros(k, dtype=np.float64) if credit is None else np.asarray(credit, dtype=np.float64)
        )
        self.source_slot = (
            np.full(k, -1, dtype=np.int32)
            if source_slot is None
            else np.asarray(source_slot, dtype=np.int32)
        )
        self.t_sim = float(t_sim)
        self.seed = int(seed)
        self.pop_max = int(pop_max)
        self.spec_version = str(spec_version)
        self.created_at = str(created_at)
        self._net: NetworkBatch | None = None

    @property
    def k(self) -> int:
        return int(self.weights.shape[0])

    @classmethod
    def load(cls, path: str) -> "ChampionEnsemble":
        d = np.load(path, allow_pickle=False)
        return cls(
            weights=d["weights"],
            fitness=d["fitness"] if "fitness" in d.files else None,
            credit=d["credit"] if "credit" in d.files else None,
            source_slot=d["source_slot"] if "source_slot" in d.files else None,
            t_sim=float(d["t_sim"]) if "t_sim" in d.files else 0.0,
            seed=int(d["seed"]) if "seed" in d.files else -1,
            pop_max=int(d["pop_max"]) if "pop_max" in d.files else 0,
            spec_version=str(d["spec_version"]) if "spec_version" in d.files else SPEC_VERSION,
            created_at=str(d["created_at"]) if "created_at" in d.files else "",
        )

    def _ensure_net(self) -> NetworkBatch:
        if self._net is None:
            self._net = NetworkBatch(self.weights)
        return self._net

    def reset(self) -> None:
        self._net = NetworkBatch(self.weights)

    def best_index(self) -> int:
        f = np.where(np.isnan(self.fitness), -np.inf, self.fitness)
        return int(np.argmax(f))

    def run_spikes(self, input_spikes: np.ndarray, *, reset: bool = True) -> np.ndarray:
        """
        input_spikes: (T, 10) in {0, 1}
        Returns: (T, K) output spikes per step, per agent.
        """
        s = np.asarray(input_spikes, dtype=np.float64)
        if s.ndim != 2 or s.shape[1] != N_INPUT:
            raise ValueError(f"input_spikes must be (T, {N_INPUT}), got {s.shape}")
        if reset:
            self.reset()
        net = self._ensure_net()
        T = s.shape[0]
        out = np.zeros((T, self.k), dtype=np.float64)
        for t in range(T):
            o = net.step(s[t])
            out[t] = o[:, 0]
        return out

    def rates_for(
        self,
        f_in_hz: float,
        duration_ms: float = 500.0,
        *,
        seed: int | None = None,
        warmup_ms: float = 0.0,
        reset: bool = True,
    ) -> np.ndarray:
        """
        Generate one Poisson stimulus and return per-agent output rate (Hz). Shape (K,).

        ``warmup_ms`` discards membrane transient before counting spikes (does not
        change the simulated input pattern; just runs and ignores those steps).
        """
        rng = np.random.default_rng(seed)
        if warmup_ms > 0:
            warm = poisson_spikes_window(rng, f_in_hz, float(warmup_ms), N_INPUT)
            self.run_spikes(warm, reset=reset)
            reset = False
        spikes = poisson_spikes_window(rng, f_in_hz, float(duration_ms), N_INPUT)
        out = self.run_spikes(spikes, reset=reset)
        secs = duration_ms / 1000.0
        return out.sum(axis=0).astype(np.float64) / secs

    def rate_for(
        self,
        f_in_hz: float,
        duration_ms: float = 500.0,
        *,
        agent: int | None = None,
        seed: int | None = None,
        warmup_ms: float = 0.0,
    ) -> float:
        """Single-agent output rate in Hz; defaults to the best champion."""
        rates = self.rates_for(f_in_hz, duration_ms, seed=seed, warmup_ms=warmup_ms)
        idx = self.best_index() if agent is None else int(agent)
        return float(rates[idx])

    def sweep(
        self,
        rates_in_hz: Iterable[float],
        duration_ms: float = 500.0,
        *,
        agent: int | None = None,
        seed: int | None = None,
        warmup_ms: float = 100.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sweep f_in over the given rates, return (rates_in, rates_out_per_request)."""
        f_in = np.array(list(rates_in_hz), dtype=np.float64)
        out = np.zeros_like(f_in)
        rng_seed = seed
        for i, fi in enumerate(f_in.tolist()):
            out[i] = self.rate_for(
                fi,
                duration_ms,
                agent=agent,
                seed=None if rng_seed is None else rng_seed + i,
                warmup_ms=warmup_ms,
            )
        return f_in, out

    def info(self) -> dict:
        return {
            "k": self.k,
            "spec_version": self.spec_version,
            "created_at": self.created_at,
            "t_sim": self.t_sim,
            "seed": self.seed,
            "pop_max": self.pop_max,
            "fitness": self.fitness.tolist(),
            "credit": self.credit.tolist(),
            "source_slot": self.source_slot.tolist(),
        }
