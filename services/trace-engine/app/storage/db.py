"""asyncpg pool. DDL is owned by Drizzle (apps/web) — Python only reads/writes.

If Postgres is unreachable at startup we log loudly and serve traces with
persist=false semantics (POST /trace with persist=true then 503s with the
fix command) rather than refusing to boot.
"""

from __future__ import annotations

import asyncio
import logging

import asyncpg

from ..config import settings

log = logging.getLogger("ets.storage")

_pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool | None:
    global _pool
    if _pool is not None:
        return _pool
    try:
        _pool = await asyncpg.create_pool(
            settings.database_url, min_size=1, max_size=4
        )
    except (OSError, asyncpg.PostgresError) as exc:
        log.warning("Postgres unavailable (%s). Run `docker-compose up -d`.", exc)
        _pool = None
        return None

    from .schema_check import verify_schema

    ok, detail = await verify_schema(_pool)
    if not ok:
        log.error("Schema check failed: %s — run `pnpm db:migrate`.", detail)
        await _pool.close()
        _pool = None
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool | None:
    return _pool


class PoolHolder:
    """Async context that acquires a connection or raises a clear error."""

    def __init__(self, pool: asyncpg.Pool | None):
        self.pool = pool

    async def __aenter__(self) -> asyncpg.Connection:
        if self.pool is None:
            raise RuntimeError("postgres-unavailable")
        self._conn = await self.pool.acquire()
        return self._conn

    async def __aexit__(self, *exc) -> None:
        if getattr(self, "_conn", None) is not None:
            await self.pool.release(self._conn)


async def with_conn():
    """Awaitable that yields a connection context, or None when DB is down."""
    pool = get_pool()
    if pool is None:
        return None
    return PoolHolder(pool)
