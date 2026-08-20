"""Startup schema verification (Drizzle is the sole DDL author).

Python never issues CREATE/ALTER. If the Drizzle migrations have not been
applied, we refuse to write and say exactly how to fix it.
"""

from __future__ import annotations

import asyncpg

EXPECTED = {
    "traces": {
        "id", "display_id", "session_id", "model_name", "model_revision",
        "device", "trace_mode", "input", "output_text", "status",
        "max_tokens", "temperature", "token_count", "duration_ms",
        "envelope", "embedding", "created_at",
    },
    "trace_events": {
        "id", "trace_id", "seq", "type", "level", "position", "layer",
        "payload", "t",
    },
    "concepts": {"id", "trace_id", "label", "score", "kind", "payload"},
    "counterfactuals": {
        "id", "trace_id", "variable", "original_word", "replacement_word",
        "prompt_text", "impact", "payload", "created_at",
    },
    "comparisons": {
        "id", "trace_id_a", "trace_id_b", "model_a", "model_b",
        "agreement", "payload", "created_at",
    },
}


async def verify_schema(pool: asyncpg.Pool) -> tuple[bool, str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = ANY($1::text[])
            """,
            list(EXPECTED.keys()),
        )
    found: dict[str, set[str]] = {}
    for row in rows:
        found.setdefault(row["table_name"], set()).add(row["column_name"])

    for table, columns in EXPECTED.items():
        if table not in found:
            return False, f"table `{table}` missing — run `pnpm db:migrate`"
        missing = columns - found[table]
        if missing:
            return False, f"table `{table}` missing columns: {sorted(missing)}"
    return True, "ok"
