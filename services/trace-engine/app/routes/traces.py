"""Trace routes: POST /trace (SSE stream), GET /trace/{id}, GET /trace/{id}/events,
POST /trace/{id}/counterfactual (spec §23), GET /trace/{id}/counterfactuals,
POST /trace/{id}/compare (spec Phase 7, V2), GET /trace/{id}/comparisons."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field

from ..config import settings
from ..engine.generate import trace_stream
from ..schemas.trace import ComparisonRequest, CounterfactualRequest, StrictModel

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
    """Lazy, lock-guarded backend load with the device self-check.

    Backends are cached PER MODEL KEY: comparison (spec Phase 7) runs one
    prompt through two registered models, and thrashing the load between
    them would cost seconds per request. gpt2-small + distilgpt2 in fp32
    is ~850 MB — both resident is the point.
    """
    from ..models.registry import MODEL_REGISTRY, set_backend_status
    from ..models.transformer_lens_backend import TransformerLensBackend, resolve_device

    key = model_key or settings.model
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {key}")

    async with _backend_lock:
        cached = _loaded_backends.get(key)
        if cached is not None:
            return cached

        if key == "fake":
            from ..models.fake_backend import FakeBackend

            backend = FakeBackend(spec)
            backend.load("fixture")
            set_backend_status(key, device="fixture", loaded=True, self_check="skipped")
        else:
            device = resolve_device(settings.device)
            backend = TransformerLensBackend(spec)
            backend.load(device)
            set_backend_status(key, device=device, loaded=True, self_check="pass")
            if settings.self_check == "on" and device != "cpu":
                ok = await asyncio.to_thread(backend.numerics_self_check)
                if not ok:
                    log.warning(
                        "MPS numerics self-check FAILED — falling back to CPU "
                        "(see TransformerLens issue #1178)"
                    )
                    backend = TransformerLensBackend(spec)
                    backend.load("cpu")
                    set_backend_status(
                        key, device="cpu", loaded=True, self_check="cpu_fallback"
                    )
        _loaded_backends[key] = backend
        return backend


_loaded_backends: dict = {}


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


# ——— counterfactuals (spec §23: WHAT WOULD CHANGE THE ANSWER?) ———————————


async def _load_complete_trace(trace_id: str) -> dict:
    """A counterfactual compares against a finished answer — load it or say why not."""
    from ..storage.db import PoolHolder, get_pool
    from ..storage.trace_reader import load_trace

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres-unavailable")
    async with PoolHolder(pool) as conn:
        trace = await load_trace(conn, trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    if trace.get("status") != "complete":
        raise HTTPException(
            status_code=400,
            detail="trace has no completed answer to compare against",
        )
    if not any(e.get("type") == "TOKEN" for e in trace["events"]):
        raise HTTPException(
            status_code=400, detail="trace emitted no tokens — nothing to compare"
        )
    return trace


def _resolve_variables(req: CounterfactualRequest, prompt: str) -> list[dict]:
    """The edited prompts this request reruns — one variable per run."""
    from ..aggregation.counterfactuals import CUSTOM_VARIABLE, applicable_substitutions

    if req.scope == "all":
        variables = applicable_substitutions(prompt)
        if not variables:
            raise HTTPException(
                status_code=400,
                detail="no dictionary words in this prompt — edit it freely "
                "(scope=prompt) instead",
            )
        return variables

    if req.scope == "one":
        applicable = applicable_substitutions(prompt)
        matches = [
            v
            for v in applicable
            if v["variable"] == req.variable and v["originalWord"] == req.originalWord
        ]
        if not matches:
            raise HTTPException(
                status_code=400, detail="variable not applicable to this prompt"
            )
        return matches

    # scope == "prompt": the free-form edit (the CounterfactualSlider spirit)
    if req.prompt is None or req.prompt.strip() == prompt.strip():
        raise HTTPException(
            status_code=400, detail="edited prompt is identical to the original"
        )
    return [
        {
            "variable": CUSTOM_VARIABLE,
            "originalWord": None,
            "replacementWord": None,
            "promptText": req.prompt,
        }
    ]


@router.post("/trace/{trace_id}/counterfactual")
async def post_counterfactual(
    trace_id: str, req: CounterfactualRequest, request: Request
) -> StreamingResponse:
    """Run counterfactuals, streaming each comparison as it completes.

    SSE, like /trace: a rerun of a 30-token answer takes seconds on the
    real model, and the browser should see each variable land when it does
    (`event: counterfactual`), then one terminal `event: done`.
    """
    from ..engine.counterfactual import run_counterfactual
    from ..engine.sse import format_sse
    from ..storage.counterfactual_store import save_counterfactual
    from ..storage.db import get_pool

    trace = await _load_complete_trace(trace_id)
    variables = _resolve_variables(req, trace["input"]["text"])
    pool = get_pool()
    if pool is None:  # _load_complete_trace already 503s; keep the guard local
        raise HTTPException(status_code=503, detail="postgres-unavailable")

    # the trace's own model answers the counterfactuals. A fresh FakeBackend
    # per request keeps fake reruns deterministic regardless of what ran
    # before (its rng is shared state); the real model is stateless greedy.
    backend = await get_backend(trace["model"]["name"])
    if backend.spec.key == "fake":
        from ..models.fake_backend import FakeBackend

        backend = FakeBackend(backend.spec)

    async def generator():
        count = 0
        for v in variables:
            result = await run_counterfactual(
                trace,
                prompt_text=v["promptText"],
                variable=v["variable"],
                original_word=v["originalWord"],
                replacement_word=v["replacementWord"],
                backend=backend,
            )
            await save_counterfactual(pool, result)
            count += 1
            yield format_sse("counterfactual", result.model_dump_json())
        yield format_sse("done", json.dumps({"traceId": trace_id, "count": count}))

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/trace/{trace_id}/counterfactuals")
async def get_counterfactuals(trace_id: str) -> dict:
    """A trace's stored counterfactuals — replay/restore for the panel."""
    from ..storage.counterfactual_store import list_counterfactuals
    from ..storage.db import PoolHolder, get_pool

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres-unavailable")
    async with PoolHolder(pool) as conn:
        exists = await conn.fetchval("SELECT 1 FROM traces WHERE id = $1", trace_id)
        if exists is None:
            raise HTTPException(status_code=404, detail="trace not found")
        results = await list_counterfactuals(conn, trace_id)
    return {"results": results}


# ——— cross-model comparison (spec Phase 7, V2 scope) ————————————————————


@router.post("/trace/{trace_id}/compare")
async def post_compare(
    trace_id: str, req: ComparisonRequest, request: Request
) -> StreamingResponse:
    """Run the anchor trace's prompt through another registered model.

    Model B's run is a full persisted trace (same pipeline, its own
    replay); the comparison is derived from both traces' TOKEN events and
    streamed as one `event: comparison` artifact, then a terminal
    `event: done`. `event: progress` frames carry B's token count while
    it answers — transport chatter, not a contract artifact.

    Token agreement is only defined under a SHARED tokenizer; anything
    else is rejected here, before any compute is spent.
    """
    from ..engine.comparison import comparison_stream
    from ..models.registry import MODEL_REGISTRY
    from ..storage.db import get_pool
    from ..storage.trace_writer import TraceWriter

    trace = await _load_complete_trace(trace_id)
    model_a = trace["model"]["name"]
    spec_a = MODEL_REGISTRY.get(model_a)
    if spec_a is None:
        raise HTTPException(
            status_code=400,
            detail=f"anchor trace's model '{model_a}' is not registered",
        )
    spec_b = MODEL_REGISTRY.get(req.model)
    if spec_b is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {req.model}")
    if spec_a.tokenizer != spec_b.tokenizer:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{model_a}' and '{req.model}' do not share a tokenizer — "
                "token ids are not comparable across them, so agreement "
                "would be a number about nothing"
            ),
        )

    pool = get_pool()
    if pool is None:  # _load_complete_trace already 503s; keep the guard local
        raise HTTPException(status_code=503, detail="postgres-unavailable")

    # a fresh FakeBackend per request keeps fake runs deterministic
    # regardless of what ran before (its rng is shared state); the real
    # models are stateless greedy
    backend = await get_backend(req.model)
    if backend.spec.key == "fake":
        from ..models.fake_backend import FakeBackend

        backend = FakeBackend(backend.spec)

    writer = TraceWriter(pool)
    writer.start()

    async def generator():
        # comparison_stream owns this writer's lifecycle (shielded stop);
        # breaking on disconnect finalizes it via the generator's finally
        async for frame in comparison_stream(
            trace, backend=backend, writer=writer, pool=pool
        ):
            if await request.is_disconnected():
                log.info("client disconnected mid-comparison")
                break
            yield frame

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/trace/{trace_id}/comparisons")
async def get_comparisons(trace_id: str) -> dict:
    """A trace's stored comparisons — replay/restore for the panel."""
    from ..storage.comparison_store import list_comparisons
    from ..storage.db import PoolHolder, get_pool

    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="postgres-unavailable")
    async with PoolHolder(pool) as conn:
        exists = await conn.fetchval("SELECT 1 FROM traces WHERE id = $1", trace_id)
        if exists is None:
            raise HTTPException(status_code=404, detail="trace not found")
        results = await list_comparisons(conn, trace_id)
    return {"results": results}
