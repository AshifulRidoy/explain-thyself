"""Trace routes: POST /trace (SSE stream), GET /trace/{id}, GET /trace/{id}/events."""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field

from ..config import settings
from ..engine.generate import trace_stream
from ..schemas.trace import StrictModel

log = logging.getLogger("ets.routes")

router = APIRouter()


class TraceRequest(StrictModel):
    prompt: str = Field(min_length=1, max_length=2000)
    model: Optional[str] = None
    traceMode: Literal["BASIC", "STANDARD", "RESEARCH"] = "STANDARD"
    maxTokens: int = Field(default=60, ge=1, le=200)
    temperature: float = Field(default=0.0, ge=0, le=2)
    topK: Optional[int] = Field(default=None, ge=1, le=200)
    seed: Optional[int] = None
    persist: bool = True


_backend_lock = asyncio.Lock()


async def get_backend(model_key: str | None):
    """Lazy, lock-guarded backend load with the device self-check."""
    from ..models.registry import MODEL_REGISTRY, set_backend_status
    from ..models.transformer_lens_backend import TransformerLensBackend, resolve_device

    key = model_key or settings.model
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {key}")

    async with _backend_lock:
        # reuse an already-loaded backend when it matches the request
        app_backend = _loaded_backend.get("instance")
        if app_backend is not None and app_backend.spec.key == key:  # type: ignore[attr-defined]
            return app_backend

        if key == "fake":
            from ..models.fake_backend import FakeBackend

            backend = FakeBackend(spec)
            backend.load("fixture")
            set_backend_status(device="fixture", loaded=True, self_check="skipped")
        else:
            device = resolve_device(settings.device)
            backend = TransformerLensBackend(spec)
            backend.load(device)
            set_backend_status(device=device, loaded=True, self_check="pass")
            if settings.self_check == "on" and device != "cpu":
                ok = await asyncio.to_thread(backend.numerics_self_check)
                if not ok:
                    log.warning(
                        "MPS numerics self-check FAILED — falling back to CPU "
                        "(see TransformerLens issue #1178)"
                    )
                    backend = TransformerLensBackend(spec)
                    backend.load("cpu")
                    set_backend_status(device="cpu", loaded=True, self_check="cpu_fallback")
        _loaded_backend["instance"] = backend
        return backend


_loaded_backend: dict = {}


@router.post("/trace")
async def post_trace(req: TraceRequest, request: Request) -> StreamingResponse:
    backend = await get_backend(req.model)

    from ..storage.db import get_pool
    from ..storage.trace_writer import TraceWriter

    writer: TraceWriter | None = None
    pool = get_pool()
    if req.persist:
        if pool is None:
            return JSONResponse(
                status_code=503,
                content={
                    "code": "postgres-unavailable",
                    "message": "Run `docker compose up -d && pnpm db:migrate`, then retry.",
                },
            )
        writer = TraceWriter(pool)
        writer.start()

    async def generator():
        try:
            stream = trace_stream(
                prompt=req.prompt,
                trace_mode=req.traceMode,
                max_tokens=req.maxTokens,
                temperature=req.temperature,
                top_k=req.topK,
                seed=req.seed,
                backend=backend,
                writer=writer,
            )
            async for frame in stream:
                if await request.is_disconnected():
                    log.info("client disconnected mid-trace")
                    break
                yield frame
        finally:
            if writer is not None:
                # shielded: a cancellation racing the stream's end must not
                # interrupt persisting the trace's terminal state
                stop = asyncio.create_task(writer.stop())
                try:
                    await asyncio.shield(stop)
                except asyncio.CancelledError:
                    # the response task is being torn down; let the writer
                    # finish detached rather than orphan the trace
                    pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/trace/{trace_id}")
async def get_trace(trace_id: str) -> dict:
    from ..storage.db import PoolHolder, get_pool

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres-unavailable")
    from ..storage.trace_reader import load_trace

    async with PoolHolder(pool) as conn:
        trace = await load_trace(conn, trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace


@router.get("/trace/{trace_id}/events")
async def get_trace_events(trace_id: str) -> dict:
    from ..storage.db import PoolHolder, get_pool

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres-unavailable")
    async with PoolHolder(pool) as conn:
        trace = await load_trace(conn, trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return {"events": trace["events"]}
