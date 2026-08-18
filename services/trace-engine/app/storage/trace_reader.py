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
