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


class SimConfigBody(BaseModel):
    seed: int = 42
    pop_max: int = Field(200, ge=1, le=20000)
    n_initial: int | None = Field(100, ge=1)
    carrying_capacity: int | None = Field(None, ge=1)
    budget_mode: Literal["none", "shared"] = "none"
    target_speed_hz: float = Field(20.0, ge=0.0, le=2000.0)


class InferenceBody(BaseModel):
    f_in_hz: float = Field(50.0, ge=0.0, le=1000.0)
    target: Literal["best", "ensemble", "random"] = "best"
    top_k: int = Field(5, ge=1, le=50)
    duration_ms: float = Field(500.0, ge=10.0, le=5000.0)
    warmup_ms: float = Field(100.0, ge=0.0, le=2000.0)


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
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
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
