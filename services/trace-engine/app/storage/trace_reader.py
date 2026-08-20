"""Read stored traces back (replay source of truth)."""

from __future__ import annotations

import json

import asyncpg


async def load_trace(conn: asyncpg.Connection, trace_id: str) -> dict | None:
    row = await conn.fetchrow("SELECT * FROM traces WHERE id = $1", trace_id)
    if row is None:
        return None
    events = await conn.fetch(
        """
        SELECT payload FROM trace_events
        WHERE trace_id = $1 ORDER BY seq ASC
        """,
        trace_id,
    )
    event_dicts = [json.loads(e["payload"]) for e in events]
    output_text = row["output_text"]

    # Preferred source: the envelope exactly as it streamed (model dims,
    # sampling, revision all preserved). Rows written before the envelope
    # column existed fall back to the flat columns. asyncpg returns jsonb
    # as JSON text (no codec registered) — hence the loads here and the
    # dumps in the writer.
    if row["envelope"] is not None:
        base = json.loads(row["envelope"])
    else:
        base = {
            "id": row["id"],
            "displayId": row["display_id"],
            "model": {
                "name": row["model_name"],
                "revision": row["model_revision"],
                "device": row["device"],
                "layerCount": 0,
                "paramCount": 0,
            },
            "input": {"text": row["input"]},
            "traceMode": row["trace_mode"],
            "sampling": {
                "maxTokens": row["max_tokens"],
                "temperature": row["temperature"],
                "topK": None,
                "seed": None,
            },
            "createdAt": row["created_at"].isoformat(),
        }

    # authoritative final state lives in columns; finishReason in the
    # OUTPUT event payload (never guess it)
    base["status"] = row["status"]
    if output_text is not None:
        finish_reason = next(
            (e.get("finishReason") for e in event_dicts if e.get("type") == "OUTPUT"),
            "stop",
        )
        base["output"] = {
            "text": output_text,
            "tokenCount": row["token_count"] or 0,
            "durationMs": row["duration_ms"] or 0,
            "finishReason": finish_reason,
        }
    base["events"] = event_dicts
    return base


async def list_recent_traces(conn: asyncpg.Connection, limit: int = 50) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id, display_id, input, status, token_count,
               duration_ms, created_at, model_name
        FROM traces ORDER BY created_at DESC LIMIT $1
        """,
        limit,
    )
    return [
        {
            "id": r["id"],
            "displayId": r["display_id"],
            "input": r["input"],
            "status": r["status"],
            "tokenCount": r["token_count"],
            "durationMs": r["duration_ms"],
            "createdAt": r["created_at"].isoformat(),
            "modelName": r["model_name"],
        }
        for r in rows
    ]


# ------------------------------------------------------------ search (V2)

# Spec §28: cosine ranking in Postgres via pgvector (`<=>` = cosine
# distance, so similarity = 1 − distance). Exact scan at this scale — an
# ANN (HNSW) index is deferred until trace count makes it honest to need.
#
# Every query filters by model_name: each backend has its own
# representation space, so a cosine across models compares unrelated
# geometries. Search ranks within one model's traces, never across.

_SEARCH_SELECT = """
    SELECT id, display_id, input, model_name, trace_mode,
           token_count, created_at,
           1 - (embedding <=> $1::vector) AS similarity
    FROM traces
    WHERE embedding IS NOT NULL AND model_name = $2
"""


def _hit(r: asyncpg.Record) -> dict:
    return {
        "traceId": r["id"],
        "displayId": r["display_id"],
        "input": r["input"],
        "similarity": round(float(r["similarity"]), 4),
        "modelName": r["model_name"],
        "traceMode": r["trace_mode"],
        "tokenCount": r["token_count"],
        "createdAt": r["created_at"].isoformat(),
    }


async def search_traces(
    conn: asyncpg.Connection,
    query_vector: list[float],
    limit: int,
    model_name: str,
) -> list[dict]:
    """Rank the given model's embedded traces against one query vector."""
    rows = await conn.fetch(
        _SEARCH_SELECT + " ORDER BY embedding <=> $1::vector LIMIT $3",
        _vector_literal(query_vector), model_name, limit,
    )
    return [_hit(r) for r in rows]


async def similar_traces(
    conn: asyncpg.Connection, trace_id: str, limit: int
) -> list[dict] | None:
    """Rank a trace against ITS OWN model's other traces. None = unknown
    trace; an empty list from a known trace means it has no embedding
    (recorded before the column existed, or the embed pass failed) — the
    caller reports that honestly rather than falling back to text
    matching."""
    source = await conn.fetchrow(
        "SELECT embedding::text AS vec, model_name FROM traces WHERE id = $1",
        trace_id,
    )
    if source is None:
        return None
    if source["vec"] is None:
        return []
    # the stored vector read back as its text form is itself a valid
    # pgvector input — one parameter, computed once
    rows = await conn.fetch(
        _SEARCH_SELECT
        + " AND id != $3 ORDER BY embedding <=> $1::vector LIMIT $4",
        source["vec"], source["model_name"], trace_id, limit,
    )
    return [_hit(r) for r in rows]


def _vector_literal(embedding: list[float]) -> str:
    from ..aggregation.search import vector_literal

    return vector_literal(embedding)


async def searchable_count(conn: asyncpg.Connection, model_name: str) -> int:
    return await conn.fetchval(
        "SELECT count(*) FROM traces WHERE embedding IS NOT NULL AND model_name = $1",
        model_name,
    )


async def traces_missing_embedding(
    conn: asyncpg.Connection, model_name: str
) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id, input FROM traces
        WHERE embedding IS NULL AND model_name = $1
        ORDER BY created_at
        """,
        model_name,
    )
    return [{"id": r["id"], "input": r["input"]} for r in rows]


async def set_embedding(
    conn: asyncpg.Connection, trace_id: str, embedding: list[float]
) -> None:
    await conn.execute(
        "UPDATE traces SET embedding = $2::vector WHERE id = $1",
        trace_id, _vector_literal(embedding),
    )
