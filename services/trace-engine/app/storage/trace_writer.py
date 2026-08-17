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

import asyncpg

log = logging.getLogger("ets.storage")

_ABORT_DETAIL = "client disconnected before completion"


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
    ) -> int:
        self._trace_id = trace_id
        self._last_seq = -1
        self._terminal = False
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """
                INSERT INTO traces
                    (id, model_name, model_revision, device, trace_mode,
                     input, status, max_tokens, temperature)
                VALUES ($1, $2, $3, $4, $5, $6, 'streaming', $7, $8)
                RETURNING display_id
                """,
                trace_id, model_name, model_revision, device, trace_mode,
                input_text, max_tokens, temperature,
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
        async with self.pool.acquire() as conn:
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
            async with self.pool.acquire() as conn:
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
            async with self.pool.acquire() as conn:
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
                await self._insert_batch(batch)
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
        conn = await self.pool.acquire()
        try:
            await conn.executemany(
                """
                INSERT INTO trace_events
                    (trace_id, seq, type, level, position, layer, payload, t)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
                """,
                batch,
            )
        except asyncio.CancelledError:
            # a query cancelled mid-flight leaves the connection mid-reset;
            # terminate it so the pool replaces it instead of handing a
            # poisoned connection to the next query
            conn.terminate()
            raise
        finally:
            await self.pool.release(conn)
