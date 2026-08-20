"""Trace persistence: Python writes, Drizzle owns the DDL.

Event inserts flow through an asyncio.Queue drained by a background task
(flush per 10 events) so database latency can never delay an SSE yield.
`payload` is the exact validated event JSON the browser also received.

`stop()` is the safety net for every ending, including a client that walks
away mid-stream: it rescues any in-flight batch cancelled by shutdown, drains
the queue, and — if the trace never reached a terminal state — marks it
aborted with a DECISION event so a replay still explains why it ended.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

import asyncpg

log = logging.getLogger("ets.storage")

_ABORT_DETAIL = "client disconnected before completion"


@asynccontextmanager
async def pooled_conn(pool: asyncpg.Pool):
    """Acquire from the pool, but never return a cancelled connection to it.

    A query cancelled mid-flight leaves the connection mid-protocol-reset;
    handing it back poisons the next acquirer (hangs or corruption).
    Terminating it makes the pool open a fresh one instead.
    """
    conn = await pool.acquire()
    try:
        yield conn
    except asyncio.CancelledError:
        conn.terminate()
        raise
    finally:
        await pool.release(conn)


class TraceWriter:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool
        self._queue: asyncio.Queue[asyncpg.Record | tuple] = asyncio.Queue(maxsize=4096)
        self._task: asyncio.Task | None = None
        self._trace_id: str | None = None
        self._last_seq: int = -1
        self._terminal = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._flush_loop(), name="trace-writer")

    async def stop(self) -> None:
        if self._task is not None:
            # cancel first so drain() can't race the flush task for queue items;
            # the loop rescues any batch whose insert the cancellation interrupted
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.drain()
        if self._trace_id is not None and not self._terminal:
            await self._abort()

    # -- writes ------------------------------------------------------------

    async def open_trace(
        self,
        *,
        trace_id: str,
        model_name: str,
        model_revision: str,
        device: str,
        trace_mode: str,
        input_text: str,
        max_tokens: int,
        temperature: float,
        embedding: list[float] | None = None,
    ) -> int:
        from ..aggregation.search import vector_literal

        # asyncpg has no vector codec: send the pgvector text literal,
        # cast server-side. None stores NULL (row = unsearchable, honestly).
        embedding_sql = (
            vector_literal(embedding) if embedding is not None else None
        )
        async with pooled_conn(self.pool) as conn:
            display_id = await conn.fetchval(
                """
                INSERT INTO traces
                    (id, model_name, model_revision, device, trace_mode,
                     input, status, max_tokens, temperature, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, 'streaming', $7, $8, $9::vector)
                RETURNING display_id
                """,
                trace_id, model_name, model_revision, device, trace_mode,
                input_text, max_tokens, temperature, embedding_sql,
            )
        # writer state flips only once the row exists: a client abort can
        # cancel this insert (a trace-dial flip does exactly that), and
        # _abort must never persist events for a trace that never landed
        self._trace_id = trace_id
        self._last_seq = -1
        self._terminal = False
        return display_id

    async def save_envelope(self, trace_id: str, envelope: dict) -> None:
        """Persist the stream-start envelope (events removed). Replay reads
        it back so model dims / sampling survive the round trip exactly —
        they are not derivable from the flat columns."""
        async with pooled_conn(self.pool) as conn:
            await conn.execute(
                "UPDATE traces SET envelope = $2 WHERE id = $1",
                trace_id, json.dumps({**envelope, "events": []}),
            )

    async def enqueue_event(self, trace_id: str, event: dict) -> None:
        if event["seq"] > self._last_seq:
            self._last_seq = event["seq"]
        await self._queue.put(
            (
                trace_id,
                event["seq"],
                event["type"],
                event["level"],
                event.get("position"),
                event.get("layer"),
                json.dumps(event),
                event["t"],
            )
        )

    async def close_trace(
        self,
        trace_id: str,
        *,
        output_text: str,
        token_count: int,
        duration_ms: int,
        status: str,
    ) -> None:
        await self.drain()
        self._terminal = True
        async with pooled_conn(self.pool) as conn:
            await conn.execute(
                """
                UPDATE traces
                SET output_text = $2, token_count = $3,
                    duration_ms = $4, status = $5
                WHERE id = $1
                """,
                trace_id, output_text, token_count, duration_ms, status,
            )

    async def record_error(self, trace_id: str, message: str) -> None:
        try:
            await self.drain()
            self._terminal = True
            async with pooled_conn(self.pool) as conn:
                await conn.execute(
                    "UPDATE traces SET status = 'error', output_text = $2 WHERE id = $1",
                    trace_id, message[:2000],
                )
        except Exception:  # noqa: BLE001 — best effort on the error path
            log.exception("failed to mark trace %s as error", trace_id)

    async def _abort(self) -> None:
        """Client vanished mid-trace: persist why it ended, replayably."""
        trace_id = self._trace_id
        assert trace_id is not None
        try:
            async with pooled_conn(self.pool) as conn:
                landed = await conn.fetchval(
                    "SELECT 1 FROM traces WHERE id = $1", trace_id
                )
            if landed is None:
                # the open insert was cancelled before committing — the
                # trace was never observable, so there is nothing to explain
                log.info("abort: trace %s never landed (open cancelled)", trace_id)
                return
            await self._queue.put(
                (
                    trace_id,
                    self._last_seq + 1,
                    "DECISION",
                    "DERIVED",
                    None,
                    None,
                    json.dumps(
                        {
                            "id": f"evt_{self._last_seq + 1:04d}",
                            "seq": self._last_seq + 1,
                            "type": "DECISION",
                            "t": 0,
                            "level": "DERIVED",
                            "decision": "aborted",
                            "detail": _ABORT_DETAIL,
                        }
                    ),
                    0,
                )
            )
            await self.drain()
            async with pooled_conn(self.pool) as conn:
                await conn.execute(
                    "UPDATE traces SET status = 'error', output_text = $2 WHERE id = $1",
                    trace_id, _ABORT_DETAIL,
                )
        except Exception:  # noqa: BLE001 — best effort on the abort path
            log.exception("failed to abort trace %s", trace_id)

    # -- background flush ---------------------------------------------------

    async def drain(self) -> None:
        """Flush everything currently queued."""
        batch: list[tuple] = []
        while not self._queue.empty():
            batch.append(self._queue.get_nowait())
        if batch:
            await self._insert_batch(batch)

    async def _flush_loop(self) -> None:
        batch: list[tuple] = []
        try:
            while True:
                batch = [await self._queue.get()]
                # coalesce what is already waiting (final safety drain happens
                # in close_trace/stop, so no timer is needed for tail latency)
                while not self._queue.empty() and len(batch) < 10:
                    batch.append(self._queue.get_nowait())
                try:
                    await self._insert_batch(batch)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — one bad batch must not
                    # kill the loop: later events (incl. the terminal ones)
                    # still deserve their chance to land
                    log.exception(
                        "event batch insert failed (%d events dropped)",
                        len(batch),
                    )
                batch = []
        except asyncio.CancelledError:
            # events already dequeued are ours to land even on shutdown
            if batch:
                try:
                    await self._insert_batch(batch)
                except Exception:  # noqa: BLE001
                    log.exception(
                        "rescue insert of %d in-flight events failed", len(batch)
                    )
            raise

    async def _insert_batch(self, batch: list[tuple]) -> None:
        async with pooled_conn(self.pool) as conn:
            await conn.executemany(
                """
                INSERT INTO trace_events
                    (trace_id, seq, type, level, position, layer, payload, t)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
                """,
                batch,
            )
