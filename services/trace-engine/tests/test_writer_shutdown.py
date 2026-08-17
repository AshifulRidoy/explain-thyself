"""Writer shutdown correctness — the two ways a stream can end badly.

Regression tests for a real bug found while verifying M0: a client that
disconnects mid-trace used to (a) lose events sitting in the flush loop's
in-flight batch and (b) orphan the trace row in `streaming` forever.
"""

from __future__ import annotations

import asyncio
import json

import asyncpg
import pytest
import pytest_asyncio

from app.storage.trace_writer import TraceWriter

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


def _event(seq: int) -> dict:
    return {
        "id": f"evt_{seq:04d}",
        "seq": seq,
        "type": "TOKEN",
        "t": 10 * seq,
        "level": "MEASURED",
        "position": seq,
        "text": "x",
    }


async def _count_events(pool: asyncpg.Pool, trace_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM trace_events WHERE trace_id = $1", trace_id
        )


async def _trace_row(pool: asyncpg.Pool, trace_id: str) -> asyncpg.Record:
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM traces WHERE id = $1", trace_id)


async def test_stop_after_complete_close_is_clean(pool, clean_trace_ids):
    """close_trace reached → stop() must not re-mark or duplicate anything."""
    writer = TraceWriter(pool)
    writer.start()
    trace_id = "tr_wtest01"
    await writer.open_trace(
        trace_id=trace_id,
        model_name="fake",
        model_revision="",
        device="fixture",
        trace_mode="STANDARD",
        input_text="hello",
        max_tokens=2,
        temperature=0.0,
    )
    for seq in range(3):
        await writer.enqueue_event(trace_id, _event(seq))
    await writer.close_trace(
        trace_id, output_text="xyz", token_count=2, duration_ms=50, status="complete"
    )
    await writer.stop()

    row = await _trace_row(pool, trace_id)
    assert row["status"] == "complete"
    assert row["token_count"] == 2
    assert await _count_events(pool, trace_id) == 3
    await _cleanup(pool, trace_id)


async def test_stop_without_close_aborts_trace(pool, clean_trace_ids):
    """Client vanished → every queued event lands + terminal state + DECISION."""
    writer = TraceWriter(pool)
    writer.start()
    trace_id = "tr_wtest02"
    await writer.open_trace(
        trace_id=trace_id,
        model_name="fake",
        model_revision="",
        device="fixture",
        trace_mode="STANDARD",
        input_text="hello",
        max_tokens=5,
        temperature=0.0,
    )
    for seq in range(4):
        await writer.enqueue_event(trace_id, _event(seq))
    await writer.stop()  # no close_trace — the abort path

    row = await _trace_row(pool, trace_id)
    assert row["status"] == "error"
    assert "disconnected" in row["output_text"]
    # 4 queued + 1 synthetic aborted DECISION, gapless seq
    assert await _count_events(pool, trace_id) == 5
    async with pool.acquire() as conn:
        last = await conn.fetchrow(
            "SELECT seq, type, payload FROM trace_events "
            "WHERE trace_id = $1 ORDER BY seq DESC LIMIT 1",
            trace_id,
        )
    assert last["seq"] == 4
    assert last["type"] == "DECISION"
    assert json.loads(last["payload"])["decision"] == "aborted"
    await _cleanup(pool, trace_id)


async def test_flush_cancelled_mid_insert_rescues_batch(pool, clean_trace_ids, monkeypatch):
    """The exact M0 failure: cancel lands during executemany — batch survives."""
    writer = TraceWriter(pool)
    trace_id = "tr_wtest03"
    await writer.open_trace(
        trace_id=trace_id,
        model_name="fake",
        model_revision="",
        device="fixture",
        trace_mode="STANDARD",
        input_text="hello",
        max_tokens=9,
        temperature=0.0,
    )

    real_insert = writer._insert_batch
    calls = 0
    inside = asyncio.Event()

    async def holding_insert(batch):
        nonlocal calls
        calls += 1
        if calls == 1:
            # park the flush-loop's first insert so the test can cancel the
            # task at exactly the moment a batch is in flight
            inside.set()
            await asyncio.sleep(3600)
        return await real_insert(batch)

    monkeypatch.setattr(writer, "_insert_batch", holding_insert)
    writer.start()
    for seq in range(6):
        await writer.enqueue_event(trace_id, _event(seq))
    await inside.wait()
    writer._task.cancel()  # cancellation arrives mid-insert, like a real disconnect

    await writer.stop()

    # 6 rescued events + the synthetic aborted DECISION
    assert await _count_events(pool, trace_id) == 7
    row = await _trace_row(pool, trace_id)
    assert row["status"] == "error"  # aborted, but fully persisted
    await _cleanup(pool, trace_id)


async def _cleanup(pool: asyncpg.Pool, trace_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM traces WHERE id = $1", trace_id)


@pytest_asyncio.fixture()
async def clean_trace_ids(pool: asyncpg.Pool):
    """Tests use fixed ids and may fail before their own cleanup — wipe first."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM traces WHERE id LIKE 'tr_wtest%'")
    yield
