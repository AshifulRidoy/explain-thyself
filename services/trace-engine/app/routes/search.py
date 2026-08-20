"""Search routes (spec §28): GET /search?q=, GET /trace/{id}/similar,
POST /search/backfill.

The query is embedded by the SAME model that recorded the traces, so a
search asks "which recorded prompts does the model represent like this
one" — never "which traces mean the same thing". /similar needs no model
at all: it reuses the trace's own stored vector.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from ..aggregation.search import SEARCH_BASIS, clamp_limit
from ..schemas.trace import BackfillReport, SearchHit, SearchResponse
from ..storage import trace_reader
from ..storage.db import PoolHolder, get_pool
from .traces import get_backend

log = logging.getLogger("ets.routes")

router = APIRouter()


def _pool_or_503():
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="database not available")
    return pool


def _response(query: str, results: list[dict], searchable: int) -> SearchResponse:
    return SearchResponse(
        query=query,
        basis=SEARCH_BASIS,
        results=[SearchHit.model_validate(r) for r in results],
        searchable=searchable,
    )


@router.get("/search")
async def search(
    q: str = Query(min_length=1, max_length=2000),
    limit: int | None = Query(default=None, ge=1, le=50),
):
    pool = _pool_or_503()
    backend = await get_backend(None)
    try:
        query_vector = await asyncio.to_thread(backend.embed_prompt, q)
    except Exception:  # noqa: BLE001 — an embeddable query is a fair ask
        log.exception("query embedding failed")
        raise HTTPException(
            status_code=422, detail="query could not be embedded"
        ) from None

    model_name = backend.spec.key  # type: ignore[attr-defined]
    async with PoolHolder(pool) as conn:
        results = await trace_reader.search_traces(
            conn, query_vector, clamp_limit(limit), model_name
        )
        searchable = await trace_reader.searchable_count(conn, model_name)
    return _response(q, results, searchable)


@router.get("/trace/{trace_id}/similar")
async def similar(
    trace_id: str,
    limit: int | None = Query(default=None, ge=1, le=50),
):
    pool = _pool_or_503()
    async with PoolHolder(pool) as conn:
        results = await trace_reader.similar_traces(conn, trace_id, clamp_limit(limit))
        if results is None:
            raise HTTPException(status_code=404, detail=f"unknown trace: {trace_id}")
        source = await conn.fetchrow(
            "SELECT input, model_name FROM traces WHERE id = $1", trace_id
        )
        # the count answers "how many were compared" — same model only
        searchable = await trace_reader.searchable_count(conn, source["model_name"])
        source_input = source["input"]
    if not results and searchable > 0:
        # known trace, no stored vector: say so instead of pretending an
        # empty corpus answered
        raise HTTPException(
            status_code=409,
            detail=(
                "trace has no stored embedding — recorded before trace "
                "search existed, or the embedding pass failed"
            ),
        )
    # query field = the source trace's prompt, so the response stands alone
    return _response(source_input or trace_id, results, searchable)


@router.post("/search/backfill")
async def backfill():
    """Re-embed traces recorded before the embedding column existed —
    for the CURRENT model only (each backend has its own representation
    space; a cosine across models compares unrelated geometries). The
    embedding depends only on prompt text and model — both stored — so
    this re-derives what generation would have written, never guesses."""
    pool = _pool_or_503()
    backend = await get_backend(None)
    model_name = backend.spec.key  # type: ignore[attr-defined]
    async with PoolHolder(pool) as conn:
        missing = await trace_reader.traces_missing_embedding(conn, model_name)
    filled = 0
    for row in missing:
        try:
            vector = await asyncio.to_thread(backend.embed_prompt, row["input"])
            async with PoolHolder(pool) as conn:
                await trace_reader.set_embedding(conn, row["id"], vector)
            filled += 1
        except Exception:  # noqa: BLE001 — one bad row must not stop the rest
            log.exception("backfill: embedding failed for trace %s", row["id"])
    async with PoolHolder(pool) as conn:
        remaining = len(
            await trace_reader.traces_missing_embedding(conn, model_name)
        )
    return BackfillReport(filled=filled, remaining=remaining)
