"""Counterfactual persistence. Drizzle owns the DDL; Python writes the
rows. `payload` jsonb is the exact validated CounterfactualResult JSON the
browser also received — replay is byte-faithful, same rule as trace_events.
"""

from __future__ import annotations

import json

import asyncpg

from ..schemas.trace import CounterfactualResult
from .trace_writer import pooled_conn


async def save_counterfactual(
    pool: asyncpg.Pool, result: CounterfactualResult
) -> None:
    """Direct insert — a counterfactual request is not the hot path, so it
    does not need the writer's queue."""
    async with pooled_conn(pool) as conn:
        await conn.execute(
            """
            INSERT INTO counterfactuals
                (id, trace_id, variable, original_word, replacement_word,
                 prompt_text, impact, payload)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            """,
            result.id,
            result.traceId,
            result.variable,
            result.originalWord,
            result.replacementWord,
            result.promptText,
            result.impact,
            json.dumps(result.model_dump()),
        )


async def list_counterfactuals(
    conn: asyncpg.Connection, trace_id: str
) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT payload FROM counterfactuals
        WHERE trace_id = $1 ORDER BY created_at ASC, id ASC
        """,
        trace_id,
    )
    return [json.loads(r["payload"]) for r in rows]
