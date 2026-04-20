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
from .neuron import N_INPUT, NetworkSingle, unpack_weights
from .population import Population
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

    def normalized(self) -> "SimConfig":
        return SimConfig(
            seed=int(self.seed),
            pop_max=max(1, int(self.pop_max)),
            n_initial=(int(self.n_initial) if self.n_initial else None),
            carrying_capacity=(int(self.carrying_capacity) if self.carrying_capacity else None),
            budget_mode=str(self.budget_mode),
            target_speed_hz=float(max(0.0, self.target_speed_hz)),
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
    ) -> dict[str, Any]:
        """Run inference on the chosen agent(s). Returns mean output Hz + per-agent details."""
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

        # Heavy work outside the lock.
        f_in = float(max(0.0, f_in_hz))
        outputs: list[float] = []
        for w in weights_per:
            net = NetworkSingle(w, rng=np.random.default_rng())
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
        }

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
            return {
                "slot": slot,
                "alive": alive,
                "credit": credit,
                "fitness": fitness,
                "topology": _agent_topology(weights),
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
