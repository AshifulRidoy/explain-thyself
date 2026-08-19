"""Python-writes → Python-reads round trip through the real Drizzle schema.

Marked `db`: requires `docker-compose up -d && pnpm db:migrate`.
The web side (Drizzle reads) is exercised by the Next.js app itself.
"""

from __future__ import annotations

import json

import asyncpg
import pytest

from app.engine.generate import new_trace_id, trace_stream
from app.models.fake_backend import FakeBackend
from app.models.registry import MODEL_REGISTRY
from app.storage.trace_writer import TraceWriter

pytestmark = pytest.mark.db


@pytest.mark.asyncio
async def test_trace_persisted_and_readable(pool: asyncpg.Pool) -> None:
    writer = TraceWriter(pool)
    writer.start()
    trace_id_holder: dict[str, str] = {}

    original = new_trace_id

    def fixed_id() -> str:
        tid = original()
        trace_id_holder["id"] = tid
        return tid

    import app.engine.generate as gen

    gen.new_trace_id = fixed_id
    try:
        async for _ in trace_stream(
            prompt="Why is the sky blue?",
            trace_mode="STANDARD",
            max_tokens=6,
            temperature=0.0,
            top_k=None,
            seed=None,
            backend=FakeBackend(MODEL_REGISTRY["fake"], seed=11),
            writer=writer,
        ):
            pass
    finally:
        gen.new_trace_id = original
        await writer.stop()

    trace_id = trace_id_holder["id"]
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM traces WHERE id = $1", trace_id)
        assert row is not None
        assert row["status"] == "complete"
        assert row["token_count"] == 6

        events = await conn.fetch(
            "SELECT payload FROM trace_events WHERE trace_id = $1 ORDER BY seq",
            trace_id,
        )
        types = [json.loads(e["payload"])["type"] for e in events]
        assert types[0] == "INPUT"
        assert "TOKEN" in types and "LAYER_ACTIVITY" in types
        assert types[-1] == "OUTPUT"
        # one token+activity pair per token, its CONCEPT events, plus
        # INPUT/DECISION/OUTPUT
        concept_count = types.count("CONCEPT")
        assert concept_count > 0, "fake STANDARD traces are concept-bearing"
        assert len(events) == 6 * 2 + 3 + concept_count
