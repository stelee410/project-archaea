"""Sweep performance / correctness tests.

Verifies that ``SimulationRuntime.sweep`` (which now batches all selected
agents into a single ``NetworkBatch`` for each (f_in, repeat) pair) produces
sane, monotone, deterministic-shape outputs that scale across many points
*and* many agents without any per-agent Python-step explosion.

The previous implementation used ``NetworkSingle`` inside three nested Python
loops (frequency × repeat × agent), making sweep wall-time scale linearly
with the number of selected agents. After the batch refactor, sweep wall-time
is dominated by the time-step loop alone — adding more agents is essentially
free. This test pins both behaviors.
"""

from __future__ import annotations

import time

import numpy as np

from archaea.runtime import SimConfig, SimulationRuntime


def _start_runtime(seed: int = 7, pop_max: int = 32) -> SimulationRuntime:
    rt = SimulationRuntime()
    rt.start(SimConfig(seed=seed, pop_max=pop_max, n_initial=pop_max, target_speed_hz=10.0))
    # Let the sim breathe a bit so that fitness gets defined for some slots.
    time.sleep(0.3)
    rt.stop()
    return rt


def test_sweep_returns_one_point_per_f_in() -> None:
    rt = _start_runtime()
    res = rt.sweep(f_in_min=10.0, f_in_max=80.0, n_points=8, target="best")
    assert res["n_points"] == 8
    assert len(res["points"]) == 8
    xs = [p["f_in_hz"] for p in res["points"]]
    assert xs == sorted(xs)
    assert all(p["n_agents"] == 1 for p in res["points"])


def test_sweep_ensemble_uses_batched_agents() -> None:
    """Ensemble with top_k=10 should report n_agents=10 in every point and
    return finite, non-negative outputs (the actual numbers depend on the
    untrained population)."""
    rt = _start_runtime(pop_max=40)
    res = rt.sweep(
        f_in_min=20.0,
        f_in_max=60.0,
        n_points=5,
        target="ensemble",
        top_k=10,
        duration_ms=200.0,
        warmup_ms=50.0,
    )
    assert all(p["n_agents"] == 10 for p in res["points"])
    for p in res["points"]:
        assert np.isfinite(p["f_out_hz_mean"])
        assert p["f_out_hz_mean"] >= 0.0
        assert len(p["f_out_hz_per_repeat"]) == 1


def test_sweep_explicit_seq_preserves_order_and_values() -> None:
    rt = _start_runtime()
    seq = [50.0, 10.0, 90.0, 30.0]
    res = rt.sweep(
        f_in_min=0.0,
        f_in_max=0.0,
        n_points=0,
        target="best",
        f_in_seq=seq,
        duration_ms=200.0,
        warmup_ms=50.0,
    )
    xs = [p["f_in_hz"] for p in res["points"]]
    assert xs == seq
    assert res["f_in_min"] == min(seq)
    assert res["f_in_max"] == max(seq)


def test_sweep_repeat_produces_per_repeat_list() -> None:
    rt = _start_runtime()
    res = rt.sweep(
        f_in_min=20.0,
        f_in_max=80.0,
        n_points=3,
        target="best",
        repeats=3,
        duration_ms=200.0,
        warmup_ms=50.0,
    )
    for p in res["points"]:
        assert len(p["f_out_hz_per_repeat"]) == 3
        # std is computed when repeats > 1
        assert p["f_out_hz_std"] >= 0.0


def test_sweep_calibration_fits_affine() -> None:
    """When calibrate=True and the response is non-degenerate, the result
    must include a finite (a, b) pair."""
    rt = _start_runtime(pop_max=40)
    res = rt.sweep(
        f_in_min=10.0,
        f_in_max=100.0,
        n_points=10,
        target="ensemble",
        top_k=10,
        duration_ms=300.0,
        warmup_ms=50.0,
        calibrate=True,
    )
    cal = res["calibration"]
    if cal["applied"]:
        assert np.isfinite(cal["a"]) and cal["a"] > 0.0
        assert np.isfinite(cal["b"])
        # Every point should have the calibrated field
        assert all("f_out_hz_calibrated" in p for p in res["points"])
    else:
        # If skipped, the reason must be one of the documented codes
        assert cal["skipped_reason"] in {
            "need_at_least_2_distinct_f_in",
            "non_monotone_or_negative_slope",
        }


def test_sweep_ensemble_scales_subliearly_in_n_agents() -> None:
    """Doubling the number of agents must NOT double sweep wall time —
    the batch refactor means time is dominated by the per-ms step loop,
    not by per-agent Python overhead.

    We compare top_k=2 vs top_k=20 on the same population/config; the
    20-agent run must finish in less than 5× the 2-agent run's time
    (a generous bound that the *old* per-agent implementation would
    blow through, since it scaled ~10×)."""
    rt = _start_runtime(pop_max=40)

    def _time_sweep(k: int) -> float:
        t0 = time.perf_counter()
        rt.sweep(
            f_in_min=10.0,
            f_in_max=100.0,
            n_points=8,
            target="ensemble",
            top_k=k,
            duration_ms=200.0,
            warmup_ms=50.0,
        )
        return time.perf_counter() - t0

    # Warm up once so JIT/import effects don't bias the first call.
    _time_sweep(2)
    t_small = _time_sweep(2)
    t_large = _time_sweep(20)
    # 10× the agents should be cheap with batching. Allow generous slack
    # for CI noise.
    assert t_large < t_small * 5.0, (
        f"sweep does not appear batched: t(k=2)={t_small:.3f}s, "
        f"t(k=20)={t_large:.3f}s, ratio={t_large / max(t_small, 1e-6):.2f}"
    )
