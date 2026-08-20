"""Comparison persistence. Drizzle owns the DDL; Python writes the rows.
`payload` jsonb is the exact validated ComparisonResult JSON the browser
also received — replay is byte-faithful, same rule as trace_events and
counterfactuals.
"""

from __future__ import annotations

import json

import asyncpg

from ..schemas.trace import ComparisonResult
from .trace_writer import pooled_conn


async def save_comparison(pool: asyncpg.Pool, result: ComparisonResult) -> None:
    """Direct insert — a comparison is not the hot path, so it does not
    need the writer's queue. The FK on trace_id_b is satisfied because
    open_trace writes B's row directly (not queued) before any frame."""
    async with pooled_conn(pool) as conn:
        await conn.execute(
            """
            INSERT INTO comparisons
                (id, trace_id_a, trace_id_b, model_a, model_b,
                 agreement, payload)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            """,
            result.id,
            result.traceIdA,
            result.traceIdB,
            result.modelA,
            result.modelB,
            result.agreement,
            json.dumps(result.model_dump()),
        )


async def list_comparisons(conn: asyncpg.Connection, trace_id: str) -> list[dict]:
    """A trace's stored comparisons, anchored on trace A."""
    rows = await conn.fetch(
        """
        SELECT payload FROM comparisons
        WHERE trace_id_a = $1 ORDER BY created_at ASC, id ASC
        """,
        trace_id,
    )
    return [json.loads(r["payload"]) for r in rows]
