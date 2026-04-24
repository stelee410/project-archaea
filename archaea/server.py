"""
FastAPI + WebSocket server for the Archaea WebUI.

Run:
    python -m archaea.server                                  # 0.0.0.0:8000
    python -m archaea.server --host 127.0.0.1 --port 8000

REST:
    GET  /api/status
    POST /api/start             body: SimConfig JSON
    POST /api/stop
    POST /api/inference         body: {f_in_hz, target, top_k, duration_ms, warmup_ms}
    POST /api/feedback          body: {slots:[...], delta_per_slot, label, f_in_hz?, f_out_hz?}
    GET  /api/agent/{slot}      within-agent 10→20→1 topology + status
    GET  /api/feedback-log?limit=N

WebSocket:
    GET /ws/telemetry           server pushes one JSON event per simulation window
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .runtime import SimConfig, get_runtime
from .strain import (
    delete_strain,
    list_strains,
)


class SimConfigBody(BaseModel):
    seed: int = 42
    pop_max: int = Field(200, ge=1, le=20000)
    n_initial: int | None = Field(100, ge=1)
    carrying_capacity: int | None = Field(None, ge=1)
    budget_mode: Literal["none", "shared"] = "none"
    target_speed_hz: float = Field(20.0, ge=0.0, le=2000.0)
    # Slime-mold extension (defaults preserve SPEC v1.0 behaviour)
    slime_mold: bool = False
    grid_size: int = Field(16, ge=4, le=128)
    pheromone_decay: float = Field(0.05, ge=0.0, le=1.0)
    pheromone_diffusion: float = Field(0.20, ge=0.0, le=1.0)
    pheromone_emit: float = Field(0.5, ge=0.0)
    pheromone_bonus_k: float = Field(0.5, ge=0.0, le=10.0)
    hgt_enabled: bool = True
    hgt_prob: float = Field(0.02, ge=0.0, le=1.0)
    hgt_blend: float = Field(0.30, ge=0.0, le=1.0)
    migrate_enabled: bool = True
    migrate_prob: float = Field(0.30, ge=0.0, le=1.0)
    # SPEC v1.2 (off-SPEC) fitness magnitude calibration penalty
    calibration_lambda: float = Field(0.0, ge=0.0, le=5.0)
    # SPEC v1.2 (off-SPEC) output-layer synaptic gain g
    synapse_gain: float = Field(1.0, gt=0.0, le=20.0)
    # SPEC_L2_V2.0 — evolution task
    task: Literal["l1", "l2v2_ctrl"] = "l1"
    # SPEC_L2_V2.0 §0 — environment shaping preset (only used for l2v2_ctrl).
    task_difficulty: Literal[
        "uniform", "balanced", "hard", "extreme", "and_only", "not_only"
    ] = "balanced"
    # SPEC_L2_V3.0 / SPEC_L2_V3.4 — admixture experiment (杂交皿).
    founders: list["FounderEntry"] | None = None
    # 3-phase ecological admixture protocol (replaces v3.0 single window):
    #   Phase 1 — commensal (HGT off):    0 .. admixture_commensal_s
    #   Phase 2 — controlled exchange:    commensal_s .. commensal_s + exchange_s
    #   Phase 3 — restored (baseline HGT):afterwards
    # Defaults of 0/0 disable the protocol entirely.
    admixture_commensal_s: float = Field(0.0, ge=0.0, le=600.0)
    admixture_exchange_s: float = Field(0.0, ge=0.0, le=1200.0)
    admixture_phase2_blend: float = Field(0.05, ge=0.0, le=1.0)
    admixture_phase2_prob_mul: float = Field(1.0, ge=0.0, le=50.0)
    # SPEC_L2_V3.5 — assortative HGT (prezygotic isolation by niche similarity).
    # null → legacy / disabled (richest neighbour wins, bit-identical to v3.4);
    # 0.0 → strict speciation (only closest-niche donor); larger values relax.
    assortative_temperature: float | None = Field(None, ge=0.0, le=10.0)


class FounderEntry(BaseModel):
    strain_id: str
    fraction: float = Field(0.5, gt=0.0, le=1.0)


class StrainSaveBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    note: str = Field("", max_length=500)


# Resolve the forward-ref FounderEntry inside SimConfigBody now that
# FounderEntry is in scope.  Pydantic v2 needs this since SimConfigBody was
# defined first (so the docstring stays at the top).
SimConfigBody.model_rebuild()


class InferenceBody(BaseModel):
    f_in_hz: float = Field(50.0, ge=0.0, le=1000.0)
    # SPEC_L2_V3.5b — niche-aware targets join the legacy quartet:
    #   colony      : route by f_s_hz (AND→and_expert, NOT→not_expert)
    #   and_expert  : top-K AND-or-DUAL specialists, ranked by acc_AND
    #   not_expert  : top-K NOT-or-DUAL specialists, ranked by acc_NOT
    #   dual_expert : DUAL specialists only (rare; mostly diagnostic)
    target: Literal[
        "best",
        "ensemble",
        "random",
        "swarm",
        "colony",
        "and_expert",
        "not_expert",
        "dual_expert",
    ] = "best"
    top_k: int = Field(5, ge=1, le=50)
    duration_ms: float = Field(500.0, ge=10.0, le=5000.0)
    warmup_ms: float = Field(100.0, ge=0.0, le=2000.0)
    swarm_radius: int = Field(1, ge=1, le=8)
    # SPEC_L2_V2.0 — optional second / selector channels.  When omitted on an
    # L2v2 sim, the runtime defaults f_b = f_in and f_s = AND instruction.
    f_b_hz: float | None = Field(None, ge=0.0, le=1000.0)
    f_s_hz: float | None = Field(None, ge=0.0, le=1000.0)


class SweepBody(BaseModel):
    f_in_min: float = Field(0.0, ge=0.0, le=1000.0)
    f_in_max: float = Field(200.0, ge=0.0, le=1000.0)
    n_points: int = Field(15, ge=2, le=64)
    # SPEC_L2_V3.5b — same expanded set as InferenceBody.  sweep does not
    # carry an f_s, so 'colony'/'auto' degrades to 'ensemble' internally.
    target: Literal[
        "best",
        "ensemble",
        "random",
        "swarm",
        "colony",
        "and_expert",
        "not_expert",
        "dual_expert",
    ] = "best"
    top_k: int = Field(5, ge=1, le=50)
    duration_ms: float = Field(500.0, ge=10.0, le=5000.0)
    warmup_ms: float = Field(100.0, ge=0.0, le=2000.0)
    swarm_radius: int = Field(1, ge=1, le=8)
    repeats: int = Field(1, ge=1, le=10)
    # Optional explicit input pattern. When provided, n_points/min/max are ignored.
    # 1..256 points, each clamped to [0, 1000] Hz server-side.
    f_in_seq: list[float] | None = Field(None, max_length=256)
    # Plan A: inference-time affine calibration (post-process; does not touch evolution).
    calibrate: bool = False


class CalibrationLambdaBody(BaseModel):
    calibration_lambda: float = Field(..., ge=0.0, le=5.0)


class SynapseGainBody(BaseModel):
    synapse_gain: float = Field(..., gt=0.0, le=20.0)


class FeedbackBody(BaseModel):
    slots: list[int]
    delta_per_slot: float = 5.0
    label: Literal["correct", "wrong", "manual"] = "manual"
    f_in_hz: float | None = None
    f_out_hz: float | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    get_runtime().attach_loop(loop)
    yield
    get_runtime().stop()


app = FastAPI(title="Project Archaea WebUI", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    return get_runtime().status()


@app.post("/api/start")
async def api_start(body: SimConfigBody) -> dict[str, Any]:
    if body.budget_mode == "shared" and not body.carrying_capacity:
        raise HTTPException(
            status_code=400,
            detail="budget_mode='shared' 需要同时填写 carrying_capacity (>0)。",
        )
    cfg = SimConfig(
        seed=body.seed,
        pop_max=body.pop_max,
        n_initial=body.n_initial,
        carrying_capacity=body.carrying_capacity,
        budget_mode=body.budget_mode,
        target_speed_hz=body.target_speed_hz,
        slime_mold=body.slime_mold,
        grid_size=body.grid_size,
        pheromone_decay=body.pheromone_decay,
        pheromone_diffusion=body.pheromone_diffusion,
        pheromone_emit=body.pheromone_emit,
        pheromone_bonus_k=body.pheromone_bonus_k,
        hgt_enabled=body.hgt_enabled,
        hgt_prob=body.hgt_prob,
        hgt_blend=body.hgt_blend,
        migrate_enabled=body.migrate_enabled,
        migrate_prob=body.migrate_prob,
        calibration_lambda=body.calibration_lambda,
        synapse_gain=body.synapse_gain,
        task=body.task,
        task_difficulty=body.task_difficulty,
        founders=(
            [{"strain_id": f.strain_id, "fraction": f.fraction} for f in body.founders]
            if body.founders
            else None
        ),
        admixture_commensal_s=body.admixture_commensal_s,
        admixture_exchange_s=body.admixture_exchange_s,
        admixture_phase2_blend=body.admixture_phase2_blend,
        admixture_phase2_prob_mul=body.admixture_phase2_prob_mul,
        assortative_temperature=body.assortative_temperature,
    )
    try:
        return get_runtime().start(cfg)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/stop")
async def api_stop() -> dict[str, Any]:
    return get_runtime().stop()


@app.post("/api/inference")
async def api_inference(body: InferenceBody) -> dict[str, Any]:
    rt = get_runtime()
    if not rt.is_running():
        raise HTTPException(status_code=409, detail="simulation not running")
    try:
        return await asyncio.to_thread(
            rt.query,
            body.f_in_hz,
            body.target,
            body.top_k,
            body.duration_ms,
            body.warmup_ms,
            body.swarm_radius,
            body.f_b_hz,
            body.f_s_hz,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/sweep")
async def api_sweep(body: SweepBody) -> dict[str, Any]:
    rt = get_runtime()
    if not rt.is_running():
        raise HTTPException(status_code=409, detail="simulation not running")
    try:
        return await asyncio.to_thread(
            rt.sweep,
            body.f_in_min,
            body.f_in_max,
            body.n_points,
            body.target,
            body.top_k,
            body.duration_ms,
            body.warmup_ms,
            body.swarm_radius,
            body.repeats,
            body.f_in_seq,
            body.calibrate,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/calibration-lambda")
async def api_set_calibration_lambda(body: CalibrationLambdaBody) -> dict[str, Any]:
    rt = get_runtime()
    if not rt.is_running():
        raise HTTPException(status_code=409, detail="simulation not running")
    try:
        return rt.set_calibration_lambda(body.calibration_lambda)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/synapse-gain")
async def api_set_synapse_gain(body: SynapseGainBody) -> dict[str, Any]:
    rt = get_runtime()
    if not rt.is_running():
        raise HTTPException(status_code=409, detail="simulation not running")
    try:
        return rt.set_synapse_gain(body.synapse_gain)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/feedback")
async def api_feedback(body: FeedbackBody) -> dict[str, Any]:
    rt = get_runtime()
    if not rt.is_running():
        raise HTTPException(status_code=409, detail="simulation not running")
    try:
        return rt.feedback(
            slots=body.slots,
            delta_per_slot=body.delta_per_slot,
            label=body.label,
            f_in_hz=body.f_in_hz,
            f_out_hz=body.f_out_hz,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/agent/{slot}")
async def api_agent(slot: int) -> dict[str, Any]:
    rt = get_runtime()
    try:
        return rt.agent_detail(slot)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/feedback-log")
async def api_feedback_log(limit: int = 100) -> list[dict[str, Any]]:
    return get_runtime().feedback_log(limit=int(limit))


# ── SPEC_L2_V3.0 — strain (菌株) endpoints ────────────────────────────────


@app.get("/api/strains")
async def api_list_strains() -> list[dict[str, Any]]:
    """Return metadata for all saved strains (cheap; reads only sidecar JSONs)."""
    return [m.to_dict() for m in list_strains()]


@app.post("/api/strains/save")
async def api_save_strain(body: StrainSaveBody) -> dict[str, Any]:
    """Snapshot the currently-living population as a new strain on disk."""
    rt = get_runtime()
    if not rt.is_running():
        raise HTTPException(status_code=409, detail="simulation not running")
    try:
        return await asyncio.to_thread(rt.snapshot_strain, name=body.name, note=body.note)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/strains/{strain_id}")
async def api_delete_strain(strain_id: str) -> dict[str, Any]:
    removed = delete_strain(strain_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"strain not found: {strain_id}")
    return {"removed": True, "id": strain_id}


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket) -> None:
    await ws.accept()
    rt = get_runtime()
    q = rt.subscribe()
    try:
        await ws.send_json({"type": "hello", "status": rt.status()})
        while True:
            event = await q.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        rt.unsubscribe(q)


# Optional: serve built React UI from webui/dist if present
_DIST = Path(__file__).resolve().parents[1] / "webui" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    async def _index() -> FileResponse:
        return FileResponse(_DIST / "index.html")

    @app.get("/{path:path}")
    async def _spa_fallback(path: str) -> FileResponse:
        candidate = _DIST / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")


def main() -> int:
    p = argparse.ArgumentParser(description="Archaea WebUI server")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    import uvicorn

    uvicorn.run(
        "archaea.server:app",
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
