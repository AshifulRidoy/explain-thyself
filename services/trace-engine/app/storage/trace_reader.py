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
    output_text = row["output_text"]
    return {
        "id": row["id"],
        "displayId": row["display_id"],
        "model": {
            "name": row["model_name"],
            "revision": row["model_revision"],
            "device": row["device"],
            "layerCount": 0,  # not stored in MVP; derived client-side from events
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
        "status": row["status"],
        "createdAt": row["created_at"].isoformat(),
        **(
            {
                "output": {
                    "text": output_text or "",
                    "tokenCount": row["token_count"] or 0,
                    "durationMs": row["duration_ms"] or 0,
                    "finishReason": "stop",
                }
            }
            if output_text is not None
            else {}
        ),
        "events": [json.loads(e["payload"]) for e in events],
    }


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
