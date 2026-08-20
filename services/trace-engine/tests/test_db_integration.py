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


@pytest.mark.asyncio
async def test_counterfactual_api_round_trip() -> None:
    """The full spec §23 loop through HTTP + Postgres: a persisted trace,
    the WHAT-WOULD-CHANGE button (scope=all → one SSE frame per variable,
    each persisted as it lands), restore via GET, and the honest rejections."""
    import httpx

    from app.schemas.trace import CounterfactualResult
    from app.storage import db

    pool = await db.init_pool()
    if pool is None:
        pytest.skip("postgres not running — docker-compose up -d")
    try:
        transport = httpx.ASGITransport(app=__import__("app.main", fromlist=["app"]).app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. a persisted fake trace with a completed answer
            resp = await client.post(
                "/trace",
                json={
                    "prompt": "Why is the sky blue?",
                    "model": "fake",
                    "maxTokens": 6,
                    "traceMode": "STANDARD",
                },
            )
            frames = [
                json.loads(line[6:])
                for line in resp.text.split("\n")
                if line.startswith("data: ")
            ]
            trace_id = frames[-1]["traceId"]

            # 2. the investigation: every applicable dictionary variable
            resp = await client.post(
                f"/trace/{trace_id}/counterfactual", json={"scope": "all"}
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            cf_frames = [
                json.loads(line[6:])
                for line in resp.text.split("\n")
                if line.startswith("data: ")
            ]
            results = [f for f in cf_frames if "variable" in f]
            done = cf_frames[-1]
            assert done["count"] == len(results) == 3  # why/sky/blue
            assert {r["variable"] for r in results} == {"question type", "subject"}
            for r in results:  # every frame is the validated contract
                CounterfactualResult.model_validate(r)
            by_word = {r["originalWord"]: r for r in results}
            assert by_word["sky"]["promptText"] == "Why is the ocean blue?"
            assert by_word["sky"]["tokenCount"] == 6
            assert 0.0 <= by_word["sky"]["impact"] <= 1.0

            # 3. restore: GET returns exactly what streamed (payload is truth)
            resp = await client.get(f"/trace/{trace_id}/counterfactuals")
            assert resp.status_code == 200
            restored = resp.json()["results"]
            assert [r["id"] for r in restored] == [r["id"] for r in results]
            assert restored[0] == results[0]

            # 4. the honest rejections
            resp = await client.post(
                f"/trace/{trace_id}/counterfactual",
                json={"scope": "prompt", "prompt": "Why is the sky blue? "},
            )
            assert resp.status_code == 400  # identical after strip
            resp = await client.post(
                f"/trace/{trace_id}/counterfactual",
                json={"scope": "one", "variable": "mood", "originalWord": "sky"},
            )
            assert resp.status_code == 400
            resp = await client.post(
                "/trace/tr_missing00/counterfactual", json={"scope": "all"}
            )
            assert resp.status_code == 404
    finally:
        await db.close_pool()


@pytest.mark.asyncio
async def test_search_api_round_trip(monkeypatch) -> None:
    """Spec §28 through HTTP + Postgres: every persisted trace stores its
    embedding at open, /search ranks by it (the fake is exact-match: same
    prompt → 1.0, others near 0), /trace/{id}/similar excludes the source
    row, backfill re-derives NULLs, and the response validates against
    the shared contract."""
    import httpx

    from app.config import settings
    from app.schemas.trace import SearchResponse
    from app.storage import db

    # /search embeds with settings.model — keep the whole round trip on
    # the fake (embedding must come from the same backend that recorded)
    monkeypatch.setattr(settings, "model", "fake")

    pool = await db.init_pool()
    if pool is None:
        pytest.skip("postgres not running — docker-compose up -d")
    try:
        transport = httpx.ASGITransport(app=__import__("app.main", fromlist=["app"]).app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            prompts = [
                "Why is the sky blue?",
                "The capital of France is",
                "Should I learn Python or Rust?",
                # unique in the corpus — its ranking can be asserted
                # without ties from historical dev traces
                "A purple quantized wombat dreams.",
            ]
            ids = []
            for text in prompts:
                resp = await client.post(
                    "/trace",
                    json={"prompt": text, "model": "fake", "maxTokens": 6, "traceMode": "STANDARD"},
                )
                assert resp.status_code == 200
                ids.append(resp.text.split('"id":"', 1)[1].split('"', 1)[0])

            # 1. every new trace row carries its embedding
            async with pool.acquire() as conn:
                stored = await conn.fetch(
                    "SELECT id, embedding IS NOT NULL AS has_vec FROM traces WHERE id = ANY($1)",
                    ids,
                )
            assert len(stored) == 4 and all(r["has_vec"] for r in stored)

            # 2. results come from the QUERY's model only — each backend
            # has its own representation space; a cosine across models
            # compares unrelated geometries (the dev corpus holds real-model
            # traces too, which must never surface in a fake-model search)
            resp = await client.get("/search", params={"q": "Why is the sky blue?", "limit": 50})
            assert resp.status_code == 200
            body = SearchResponse.model_validate(resp.json())
            assert all(h.modelName == "fake" for h in body.results)

            # 3. search ranks the same prompt at 1.0. The dev corpus holds
            # older traces of these prompts too (ties at 1.0 are honest) —
            # so assert on OUR row's score, not on it being literally first.
            resp = await client.get("/search", params={"q": "Why is the sky blue?", "limit": 50})
            assert resp.status_code == 200
            body = SearchResponse.model_validate(resp.json())
            assert body.results[0].input == "Why is the sky blue?"
            assert body.results[0].similarity == 1.0
            ours = next(h for h in body.results if h.traceId == ids[0])
            assert ours.similarity == 1.0
            assert body.searchable >= 4
            assert "not semantic meaning" in body.basis

            # the unique prompt tops its own search (ties with older runs
            # of this very test are honest), and every OTHER distinct
            # prompt scores near 0 — the fake's vectors are centered, so
            # unrelated really means unrelated
            resp = await client.get(
                "/search", params={"q": "A purple quantized wombat dreams.", "limit": 50}
            )
            body = SearchResponse.model_validate(resp.json())
            assert body.results[0].similarity == 1.0
            assert body.results[0].input == "A purple quantized wombat dreams."
            ours_wombat = next(h for h in body.results if h.traceId == ids[3])
            assert ours_wombat.similarity == 1.0
            distinct = {r.input: r.similarity for r in body.results}
            # every OTHER distinct prompt scores near 0 — the fake's
            # vectors are centered, so unrelated really means unrelated
            for text, sim in distinct.items():
                if text != "A purple quantized wombat dreams.":
                    assert sim < 0.5, (text, sim)

            # 4. similar: ranked, source row excluded, contract-validated
            resp = await client.get(f"/trace/{ids[0]}/similar", params={"limit": 5})
            assert resp.status_code == 200
            sim = SearchResponse.model_validate(resp.json())
            assert all(h.traceId != ids[0] for h in sim.results)
            assert sim.query == "Why is the sky blue?"  # response stands alone

            # 5. honest rejections
            resp = await client.get("/trace/tr_nope0000/similar")
            assert resp.status_code == 404
            resp = await client.get("/search", params={"q": "x", "limit": 99})
            assert resp.status_code == 422  # limit is bounded; it is not a hint

            # 6. backfill re-derives a NULL embedding from stored columns
            async with pool.acquire() as conn:
                await conn.execute("UPDATE traces SET embedding = NULL WHERE id = $1", ids[1])
            resp = await client.post("/search/backfill")
            assert resp.status_code == 200
            report = resp.json()
            assert report["filled"] >= 1 and report["remaining"] == 0
            async with pool.acquire() as conn:
                restored = await conn.fetchval(
                    "SELECT embedding IS NOT NULL FROM traces WHERE id = $1", ids[1]
                )
            assert restored is True
    finally:
        await db.close_pool()
