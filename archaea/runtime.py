"""
Thread-safe live simulation runtime for the WebUI backend.

A single ``SimulationRuntime`` instance is owned by ``archaea.server`` (one sim
per process).  The simulation runs in a daemon thread and publishes telemetry
events into per-client asyncio queues via ``loop.call_soon_threadsafe``.

This is **off-SPEC**: there is no auto-halt, no checkpointing.  The CLI ``run.py``
remains the canonical SPEC-compliant runner.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .economy import BUDGET_MODE_NONE, R_MAX
from .neuron import N_INPUT, NetworkBatch, NetworkSingle, unpack_weights
from .population import Population
from .slime import SlimeConfig
from .stimulus import poisson_spikes_window

WINDOW_S = 0.5


@dataclass
class SimConfig:
    seed: int = 42
    pop_max: int = 200
    n_initial: int | None = 100
    carrying_capacity: int | None = None
    budget_mode: str = BUDGET_MODE_NONE
    target_speed_hz: float = 20.0
    """How many simulation windows per wall-clock second (0 = full speed)."""
    # Slime-mold extension (off by default — preserves SPEC v1.0 behavior bit-identical)
    slime_mold: bool = False
    grid_size: int = 16
    pheromone_decay: float = 0.05
    pheromone_diffusion: float = 0.20
    pheromone_emit: float = 0.5
    pheromone_bonus_k: float = 0.5
    hgt_enabled: bool = True
    hgt_prob: float = 0.02
    hgt_blend: float = 0.30
    migrate_enabled: bool = True
    migrate_prob: float = 0.30
    # SPEC v1.2 (off-SPEC) — fitness magnitude calibration penalty (λ).
    # 0.0 = pure Pearson r (SPEC §4.1, affine-invariant; outputs may be compressed).
    # > 0  = subtract λ · |mean(f_out) - mean(f_in)| / std(f_in) from r each window,
    #        pushing the swarm toward slope ≈ 1.
    calibration_lambda: float = 0.0
    # SPEC v1.2 (off-SPEC) — output-layer synaptic gain g.
    # 1.0 = SPEC §1.1 bit-identical. > 1 multiplies I_o, raising raw f_out by
    # making the output neuron physically spike more (within the LIF refractory limit).
    synapse_gain: float = 1.0

    def normalized(self) -> "SimConfig":
        return SimConfig(
            seed=int(self.seed),
            pop_max=max(1, int(self.pop_max)),
            n_initial=(int(self.n_initial) if self.n_initial else None),
            carrying_capacity=(int(self.carrying_capacity) if self.carrying_capacity else None),
            budget_mode=str(self.budget_mode),
            target_speed_hz=float(max(0.0, self.target_speed_hz)),
            slime_mold=bool(self.slime_mold),
            grid_size=max(4, int(self.grid_size)),
            pheromone_decay=float(self.pheromone_decay),
            pheromone_diffusion=float(self.pheromone_diffusion),
            pheromone_emit=float(self.pheromone_emit),
            pheromone_bonus_k=float(self.pheromone_bonus_k),
            hgt_enabled=bool(self.hgt_enabled),
            hgt_prob=float(self.hgt_prob),
            hgt_blend=float(self.hgt_blend),
            migrate_enabled=bool(self.migrate_enabled),
            migrate_prob=float(self.migrate_prob),
            calibration_lambda=float(max(0.0, self.calibration_lambda)),
            synapse_gain=float(max(1e-3, self.synapse_gain)),
        )

    def to_slime_config(self) -> SlimeConfig:
        return SlimeConfig(
            enabled=bool(self.slime_mold),
            grid_size=int(self.grid_size),
            pheromone_decay=float(self.pheromone_decay),
            pheromone_diffusion=float(self.pheromone_diffusion),
            pheromone_emit=float(self.pheromone_emit),
            pheromone_bonus_k=float(self.pheromone_bonus_k),
            hgt_enabled=bool(self.hgt_enabled),
            hgt_prob=float(self.hgt_prob),
            hgt_blend=float(self.hgt_blend),
            migrate_enabled=bool(self.migrate_enabled),
            migrate_prob=float(self.migrate_prob),
        )


@dataclass
class TelemetryEvent:
    type: str = "telemetry"
    t_sim: float = 0.0
    pop_size: int = 0
    pop_max: int = 0
    births: int = 0
    deaths: int = 0
    r_max: float = 0.0
    r_mean: float = 0.0
    credit_mean: float = 0.0
    credit_gini: float = 0.0
    weight_std: float = 0.0
    sigma: float = 0.0
    budget_pressure: float = 0.0
    alive: list[int] = field(default_factory=list)
    credit: list[float] = field(default_factory=list)
    fitness: list[float] = field(default_factory=list)
    dead_slots: list[int] = field(default_factory=list)
    repro_parent_slots: list[int] = field(default_factory=list)
    repro_child_slots: list[int] = field(default_factory=list)


def _bool_to_int_list(arr: np.ndarray) -> list[int]:
    return arr.astype(np.int8).tolist()


def _agent_topology(weights_220: np.ndarray) -> dict[str, Any]:
    """Return the within-agent 10→20→1 weighted graph for the inspector panel."""
    w1, w2 = unpack_weights(weights_220)
    w1 = w1.reshape(10, 20).astype(float)
    w2 = w2.reshape(20, 1).astype(float)
    edges_ih = []
    for i in range(10):
        for j in range(20):
            edges_ih.append({"src": f"i{i}", "dst": f"h{j}", "w": float(w1[i, j])})
    edges_ho = []
    for j in range(20):
        edges_ho.append({"src": f"h{j}", "dst": "o0", "w": float(w2[j, 0])})
    return {
        "input_nodes": [f"i{i}" for i in range(10)],
        "hidden_nodes": [f"h{j}" for j in range(20)],
        "output_nodes": ["o0"],
        "edges_ih": edges_ih,
        "edges_ho": edges_ho,
    }


class SimulationRuntime:
    """Single-instance background simulation. All public methods are thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pop: Population | None = None
        self._rng: np.random.Generator | None = None
        self._inference_rng = np.random.default_rng(0xC0FFEE)
        self._config: SimConfig | None = None
        self._t_sim: float = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._last_event: dict[str, Any] | None = None
        self._is_running = False

        self._feedback_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ status

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self._is_running,
                "config": (self._config.__dict__ if self._config else None),
                "t_sim": self._t_sim,
                "n_living": (int(self._pop.n_living()) if self._pop else 0),
                "pop_max": (int(self._pop.pop_max) if self._pop else 0),
                "subscribers": len(self._subscribers),
                "last_event": self._last_event,
                "feedback_count": len(self._feedback_log),
            }

    # --------------------------------------------------------------- start/stop

    def start(self, config: SimConfig) -> dict[str, Any]:
        cfg = config.normalized()
        with self._lock:
            if self._is_running:
                self._stop_locked()
            self._config = cfg
            self._rng = np.random.default_rng(cfg.seed)
            self._pop = Population(
                cfg.pop_max,
                self._rng,
                n_initial=cfg.n_initial,
                carrying_capacity=cfg.carrying_capacity,
                budget_mode=cfg.budget_mode,
                slime=cfg.to_slime_config(),
                calibration_lambda=cfg.calibration_lambda,
                synapse_gain=cfg.synapse_gain,
            )
            self._t_sim = 0.0
            self._last_event = None
            self._feedback_log.clear()
            self._stop_event.clear()
            self._is_running = True
            self._thread = threading.Thread(
                target=self._run_loop, name="archaea-sim", daemon=True
            )
            self._thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_locked()
        return self.status()

    def set_calibration_lambda(self, value: float) -> dict[str, Any]:
        """Live-update the fitness calibration penalty (no restart needed).
        Affects future fitness evaluations and all reward calculations."""
        v = float(max(0.0, value))
        with self._lock:
            if self._pop is None:
                raise RuntimeError("simulation not initialized")
            self._pop.calibration_lambda = v
            if self._config is not None:
                self._config.calibration_lambda = v
        return {"calibration_lambda": v}

    def set_synapse_gain(self, value: float) -> dict[str, Any]:
        """Live-update the output-layer synaptic gain g (no restart needed).
        Takes effect from the next simulation window AND the next inference call."""
        v = float(max(1e-3, value))
        with self._lock:
            if self._pop is None:
                raise RuntimeError("simulation not initialized")
            self._pop.synapse_gain = v
            if self._config is not None:
                self._config.synapse_gain = v
        return {"synapse_gain": v}

    def _stop_locked(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._is_running = False
        self._thread = None
        if thread is not None and thread.is_alive():
            self._lock.release()
            try:
                thread.join(timeout=2.0)
            finally:
                self._lock.acquire()

    # -------------------------------------------------------------- subscribe

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        with self._lock:
            self._subscribers.add(q)
            snapshot = self._last_event
        if snapshot is not None:
            try:
                q.put_nowait(snapshot)
            except asyncio.QueueFull:
                pass
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    # -------------------------------------------------------------- inference

    def query(
        self,
        f_in_hz: float,
        target: str = "best",
        top_k: int = 1,
        duration_ms: float = 500.0,
        warmup_ms: float = 100.0,
        swarm_radius: int = 1,
    ) -> dict[str, Any]:
        """Run inference on the chosen agent(s). Returns mean output Hz + per-agent details.

        target='swarm' (slime mode only): pick all living agents within Chebyshev
        radius ``swarm_radius`` of the strongest pheromone cell — i.e. let the
        currently-densest "foraging tendril" answer. Gracefully degrades to
        top_k-by-fitness if slime is off / pheromone field empty / hotspot empty.
        """
        swarm_hotspot: list[int] | None = None
        swarm_radius_used: int | None = None
        swarm_degraded: str | None = None
        with self._lock:
            if self._pop is None:
                raise RuntimeError("simulation not initialized")
            pop = self._pop
            if target == "best":
                slots = pop.top_k_slots(1, by="fitness").tolist()
            elif target == "ensemble":
                slots = pop.top_k_slots(max(1, int(top_k)), by="fitness").tolist()
            elif target == "random":
                idx = pop.living_indices()
                if idx.size == 0:
                    slots = []
                else:
                    slots = [int(self._inference_rng.choice(idx))]
            elif target == "swarm":
                slots, swarm_hotspot, swarm_radius_used, swarm_degraded = (
                    self._select_swarm_slots(pop, max(1, int(top_k)), int(swarm_radius))
                )
            else:
                raise ValueError(f"unknown target: {target}")
            if not slots:
                raise RuntimeError("no living agents to query")
            weights_per = [pop.weights[int(s)].copy() for s in slots]
            fitness_per = []
            for s in slots:
                if pop._fitness_defined(int(s)):
                    fitness_per.append(float(pop._fitness_slot(int(s))))
                else:
                    fitness_per.append(float("nan"))
            inference_gain = float(pop.synapse_gain)

        # Heavy work outside the lock.
        f_in = float(max(0.0, f_in_hz))
        outputs: list[float] = []
        for w in weights_per:
            net = NetworkSingle(
                w, rng=np.random.default_rng(), output_gain=inference_gain
            )
            if warmup_ms > 0:
                warmup_spikes = poisson_spikes_window(
                    self._inference_rng, f_in, float(warmup_ms), N_INPUT
                )
                for t in range(warmup_spikes.shape[0]):
                    net.step(warmup_spikes[t])
            spikes = poisson_spikes_window(
                self._inference_rng, f_in, float(duration_ms), N_INPUT
            )
            cnt = 0
            for t in range(spikes.shape[0]):
                if net.step(spikes[t]) > 0.0:
                    cnt += 1
            outputs.append(cnt / (duration_ms / 1000.0))

        f_out_mean = float(np.mean(outputs)) if outputs else 0.0
        return {
            "f_in_hz": f_in,
            "f_out_hz": f_out_mean,
            "target": target,
            "duration_ms": duration_ms,
            "warmup_ms": warmup_ms,
            "agents": [
                {
                    "slot": int(s),
                    "fitness": fitness_per[i],
                    "f_out_hz": float(outputs[i]),
                }
                for i, s in enumerate(slots)
            ],
            # Swarm-mode metadata (None for other targets)
            "swarm_hotspot": swarm_hotspot,
            "swarm_radius_used": swarm_radius_used,
            "swarm_size": (len(slots) if target == "swarm" else None),
            "swarm_degraded": swarm_degraded,
            "synapse_gain": inference_gain,
        }

    def sweep(
        self,
        f_in_min: float,
        f_in_max: float,
        n_points: int,
        target: str = "best",
        top_k: int = 1,
        duration_ms: float = 500.0,
        warmup_ms: float = 100.0,
        swarm_radius: int = 1,
        repeats: int = 1,
        f_in_seq: list[float] | None = None,
        calibrate: bool = False,
    ) -> dict[str, Any]:
        """Run inference at a list of f_in values, in order. Each point is
        repeated ``repeats`` times and averaged. Returns a list of
        {f_in_hz, f_out_hz_mean, f_out_hz_per_repeat}.

        If ``f_in_seq`` is provided, it is used verbatim (any pattern: ramp,
        wave, manual). Otherwise points are uniformly sampled in
        [f_in_min, f_in_max].

        Agents are snapshotted ONCE at the start (under the population lock)
        and reused for every f_in / repeat. The actual SNN simulation runs
        WITHOUT holding the lock, so the background sim thread is undisturbed
        and the sweep cannot deadlock against a fast / full-speed sim loop.
        """
        rep = max(1, int(repeats))
        if f_in_seq is not None and len(f_in_seq) >= 1:
            f_ins = [float(max(0.0, x)) for x in f_in_seq]
            lo = float(min(f_ins))
            hi = float(max(f_ins))
        else:
            n = max(2, int(n_points))
            lo = float(min(f_in_min, f_in_max))
            hi = float(max(f_in_min, f_in_max))
            f_ins = np.linspace(lo, hi, n).tolist()

        # ── Step 1: ONE-SHOT snapshot under the lock ─────────────────────────
        meta_hotspot: list[int] | None = None
        meta_radius: int | None = None
        meta_degraded: str | None = None
        with self._lock:
            if self._pop is None:
                raise RuntimeError("simulation not initialized")
            pop = self._pop
            if target == "best":
                slots = pop.top_k_slots(1, by="fitness").tolist()
            elif target == "ensemble":
                slots = pop.top_k_slots(max(1, int(top_k)), by="fitness").tolist()
            elif target == "random":
                idx = pop.living_indices()
                if idx.size == 0:
                    slots = []
                else:
                    slots = [int(self._inference_rng.choice(idx))]
            elif target == "swarm":
                slots, meta_hotspot, meta_radius, meta_degraded = (
                    self._select_swarm_slots(pop, max(1, int(top_k)), int(swarm_radius))
                )
            else:
                raise ValueError(f"unknown target: {target}")
            if not slots:
                raise RuntimeError("no living agents to query")
            weights_per = [pop.weights[int(s)].copy() for s in slots]
            inference_gain = float(pop.synapse_gain)
        n_agents = len(slots)
        meta_size_first = n_agents if target == "swarm" else None

        # ── Step 2: All SNN work OUTSIDE the lock ────────────────────────────
        # Vectorize across the agent dimension: build one NetworkBatch holding
        # all selected agents and step it once per ms (instead of N_AGENTS × T
        # per-agent Python calls). For the large branches (ensemble/swarm) this
        # is a 10× – 50× speedup; for best (A=1) it is a no-op.
        weights_stack = np.stack(weights_per, axis=0)  # (A, 220)
        duration_s = float(duration_ms) / 1000.0
        warmup_steps = int(round(float(warmup_ms))) if warmup_ms > 0 else 0
        duration_steps = int(round(float(duration_ms)))

        points: list[dict[str, Any]] = []
        for f in f_ins:
            f_in = float(max(0.0, f))
            outs_mean: list[float] = []  # mean across agents, per repeat
            for _ in range(rep):
                # Reset shared batch state for this (f, repeat).
                net = NetworkBatch(
                    weights_stack,
                    rng=np.random.default_rng(),
                    output_gain=inference_gain,
                )
                if warmup_steps > 0:
                    wsp = poisson_spikes_window(
                        self._inference_rng, f_in, float(warmup_ms), N_INPUT
                    )
                    for t in range(warmup_steps):
                        net.step(wsp[t])
                spikes = poisson_spikes_window(
                    self._inference_rng, f_in, float(duration_ms), N_INPUT
                )
                # Per-agent spike counts across the measurement window.
                counts = np.zeros(n_agents, dtype=np.int64)
                for t in range(duration_steps):
                    out_spikes = net.step(spikes[t])  # (A, 1)
                    counts += (out_spikes[:, 0] > 0.0).astype(np.int64)
                per_agent_hz = counts.astype(np.float64) / duration_s  # (A,)
                outs_mean.append(float(per_agent_hz.mean()))
            points.append(
                {
                    "f_in_hz": f_in,
                    "f_out_hz_mean": float(np.mean(outs_mean)),
                    "f_out_hz_std": float(np.std(outs_mean)) if rep > 1 else 0.0,
                    "f_out_hz_per_repeat": outs_mean,
                    "n_agents": n_agents,
                }
            )

        # ── Plan A: inference-time affine calibration ────────────────────────
        # Fit  y = a·x + b  via least squares on the observed (f_in, f_out)
        # samples. Only meaningful with ≥ 2 distinct f_in values *and* a > 0
        # (else there is no monotone relationship to invert).
        cal_a: float | None = None
        cal_b: float | None = None
        cal_skipped: str | None = None
        if calibrate:
            xs = np.array([p["f_in_hz"] for p in points], dtype=np.float64)
            ys = np.array([p["f_out_hz_mean"] for p in points], dtype=np.float64)
            if xs.size < 2 or float(np.var(xs)) < 1e-9:
                cal_skipped = "need_at_least_2_distinct_f_in"
            else:
                # least squares y = a*x + b
                a, b = np.polyfit(xs, ys, 1)
                if not np.isfinite(a) or a <= 0.0:
                    cal_skipped = "non_monotone_or_negative_slope"
                else:
                    cal_a = float(a)
                    cal_b = float(b)
                    for p in points:
                        cal = (p["f_out_hz_mean"] - cal_b) / cal_a
                        p["f_out_hz_calibrated"] = float(max(0.0, cal))

        return {
            "target": target,
            "n_points": len(f_ins),
            "repeats": rep,
            "duration_ms": duration_ms,
            "warmup_ms": warmup_ms,
            "f_in_min": lo,
            "f_in_max": hi,
            "points": points,
            "swarm_hotspot": meta_hotspot,
            "swarm_radius_used": meta_radius,
            "swarm_size_first": meta_size_first,
            "swarm_degraded": meta_degraded,
            "calibration": {
                "applied": cal_a is not None,
                "a": cal_a,
                "b": cal_b,
                "skipped_reason": cal_skipped,
            },
            "synapse_gain": inference_gain,
        }

    @staticmethod
    def _select_swarm_slots(
        pop: Population, top_k: int, swarm_radius: int
    ) -> tuple[list[int], list[int] | None, int | None, str | None]:
        """
        Pick the agents inside the strongest pheromone hotspot.

        Returns (slots, hotspot_xy_or_None, radius_used_or_None, degradation_reason).
        Falls back to top_k_by_fitness if slime is off / field empty / hotspot empty.
        """
        if not pop.slime.enabled:
            slots = pop.top_k_slots(max(1, top_k), by="fitness").tolist()
            return slots, None, None, "slime_disabled"
        P = pop.pheromone
        if float(P.max()) <= 0.0:
            slots = pop.top_k_slots(max(1, top_k), by="fitness").tolist()
            return slots, None, None, "pheromone_empty"
        G = int(pop.slime.grid_size)
        flat = int(np.argmax(P))
        cx, cy = int(flat // G), int(flat % G)
        R = max(1, swarm_radius)
        positions = pop.positions
        # Toroidal Chebyshev distance ≤ R
        dx = (positions[:, 0] - cx + G // 2) % G - G // 2
        dy = (positions[:, 1] - cy + G // 2) % G - G // 2
        in_hot = (np.abs(dx) <= R) & (np.abs(dy) <= R) & pop.alive
        cand = np.flatnonzero(in_hot).tolist()
        if not cand:
            slots = pop.top_k_slots(max(1, top_k), by="fitness").tolist()
            return slots, [cx, cy], R, "hotspot_empty"
        return [int(s) for s in cand], [cx, cy], R, None

    def feedback(
        self,
        slots: list[int],
        delta_per_slot: float,
        label: str,
        f_in_hz: float | None = None,
        f_out_hz: float | None = None,
    ) -> dict[str, Any]:
        """Apply credit delta to listed slots. Returns per-slot result + new credit."""
        results = []
        with self._lock:
            if self._pop is None:
                raise RuntimeError("simulation not initialized")
            pop = self._pop
            for s in slots:
                s = int(s)
                if not (0 <= s < pop.pop_max) or not bool(pop.alive[s]):
                    results.append({"slot": s, "alive": False, "applied": 0.0, "credit": 0.0, "killed": False})
                    continue
                old = float(pop.credit[s])
                new = max(0.0, old + float(delta_per_slot))
                pop.credit[s] = new
                killed = False
                if new <= 0.0:
                    pop.alive[s] = False
                    pop._reset_membrane_slot(s)
                    killed = True
                results.append(
                    {
                        "slot": s,
                        "alive": (not killed),
                        "applied": float(delta_per_slot),
                        "credit": float(new),
                        "killed": killed,
                    }
                )
            self._feedback_log.append(
                {
                    "t_sim": self._t_sim,
                    "label": label,
                    "delta_per_slot": float(delta_per_slot),
                    "slots": list(slots),
                    "f_in_hz": f_in_hz,
                    "f_out_hz": f_out_hz,
                    "wall": time.time(),
                }
            )
            if len(self._feedback_log) > 500:
                self._feedback_log = self._feedback_log[-500:]
        return {"results": results}

    # ----------------------------------------------------------------- detail

    def agent_detail(self, slot: int) -> dict[str, Any]:
        with self._lock:
            if self._pop is None:
                raise RuntimeError("simulation not initialized")
            pop = self._pop
            slot = int(slot)
            if not (0 <= slot < pop.pop_max):
                raise ValueError("slot out of range")
            alive = bool(pop.alive[slot])
            credit = float(pop.credit[slot])
            defined = pop._fitness_defined(slot) if alive else False
            fitness = float(pop._fitness_slot(slot)) if defined else None
            weights = pop.weights[slot].copy() if alive else np.zeros(220)
            position = (
                [int(pop.positions[slot, 0]), int(pop.positions[slot, 1])]
                if pop.slime.enabled
                else None
            )
            local_pheromone = (
                float(pop.pheromone[pop.positions[slot, 0], pop.positions[slot, 1]])
                if pop.slime.enabled and alive
                else None
            )
            return {
                "slot": slot,
                "alive": alive,
                "credit": credit,
                "fitness": fitness,
                "topology": _agent_topology(weights),
                "position": position,
                "local_pheromone": local_pheromone,
            }

    def feedback_log(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._feedback_log[-limit:])

    # ------------------------------------------------------------- main loop

    def _run_loop(self) -> None:
        """Daemon thread: step the sim and publish telemetry."""
        cfg = self._config
        if cfg is None:
            return
        target_dt = (1.0 / cfg.target_speed_hz) if cfg.target_speed_hz > 0 else 0.0
        next_tick = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if self._pop is None:
                        break
                    info = self._pop.step_window()
                    self._t_sim += WINDOW_S
                    n_alive = int(self._pop.alive.sum())
                    cmean = float(self._pop.credit_mean()) if n_alive else 0.0
                    cgini = float(self._pop.credit_gini()) if n_alive else 0.0
                    wstd = float(self._pop.weight_diversity_metric()) if n_alive > 1 else 0.0
                    pop_max = int(self._pop.pop_max)
                    alive_list = _bool_to_int_list(self._pop.alive)
                    credit_list = self._pop.credit.tolist()
                    fitness_list = []
                    for s in range(pop_max):
                        if self._pop.alive[s] and self._pop._fitness_defined(s):
                            fitness_list.append(float(self._pop._fitness_slot(s)))
                        else:
                            fitness_list.append(-1.0)
                    slime_enabled = bool(self._pop.slime.enabled)
                    if slime_enabled:
                        positions_list = self._pop.positions.tolist()
                        pheromone_grid = self._pop.pheromone.tolist()
                        grid_size = int(self._pop.slime.grid_size)
                    else:
                        positions_list = []
                        pheromone_grid = []
                        grid_size = 0
                    event: dict[str, Any] = {
                        "type": "telemetry",
                        "t_sim": float(self._t_sim),
                        "pop_size": n_alive,
                        "pop_max": pop_max,
                        "births": int(info["births"]),
                        "deaths": int(info["deaths"]),
                        "r_max": float(info["r_max"]),
                        "r_mean": float(info["r_mean"]),
                        "credit_mean": cmean,
                        "credit_gini": cgini,
                        "weight_std": wstd,
                        "sigma": float(info["sigma"]),
                        "budget_pressure": float(info.get("budget_pressure", 0.0)),
                        "alive": alive_list,
                        "credit": credit_list,
                        "fitness": fitness_list,
                        "dead_slots": info["dead_slots"].tolist(),
                        "repro_parent_slots": info["repro_parent_slots"].tolist(),
                        "repro_child_slots": info["repro_child_slots"].tolist(),
                        "slime_enabled": slime_enabled,
                        "grid_size": grid_size,
                        "positions": positions_list,
                        "pheromone": pheromone_grid,
                        "pheromone_max": float(info.get("pheromone_max", 0.0)),
                        "pheromone_mean": float(info.get("pheromone_mean", 0.0)),
                        "hgt_count": int(info.get("hgt_count", 0)),
                        "migrations": int(info.get("migrations", 0)),
                    }
                    self._last_event = event
                    subs = list(self._subscribers)
                    loop = self._loop
            except Exception as e:
                # Surface and stop cleanly
                err_event = {"type": "error", "message": str(e)}
                with self._lock:
                    subs = list(self._subscribers)
                    loop = self._loop
                    self._is_running = False
                self._dispatch(loop, subs, err_event)
                break

            self._dispatch(loop, subs, event)

            if target_dt > 0.0:
                next_tick += target_dt
                sleep_for = next_tick - time.perf_counter()
                if sleep_for > 0:
                    if self._stop_event.wait(timeout=sleep_for):
                        break
                else:
                    next_tick = time.perf_counter()
            else:
                # Full-speed mode: yield the GIL so concurrent /api/sweep,
                # /api/inference, /api/status etc. don't get starved.
                time.sleep(0)

        with self._lock:
            self._is_running = False

    @staticmethod
    def _dispatch(
        loop: asyncio.AbstractEventLoop | None,
        subs: list[asyncio.Queue],
        event: dict[str, Any],
    ) -> None:
        if loop is None or not subs:
            return
        for q in subs:
            try:
                loop.call_soon_threadsafe(_safe_put, q, event)
            except RuntimeError:
                pass


def _safe_put(q: asyncio.Queue, event: dict[str, Any]) -> None:
    if q.full():
        try:
            _ = q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        pass


# Module-level singleton
_RUNTIME = SimulationRuntime()


def get_runtime() -> SimulationRuntime:
    return _RUNTIME
