"""Shared fixtures: locate the committed fixture corpus (owned by the
TypeScript schema package) so BOTH runtimes validate the same files, and an
asyncpg pool for db-marked tests (requires docker-compose up + migrate)."""

from __future__ import annotations

import json
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES_DIR = REPO_ROOT / "packages" / "trace-schema" / "fixtures"

DB_URL = "postgresql://ets:ets@localhost:5432/explain_the_self"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    if not FIXTURES_DIR.is_dir():
        pytest.fail(f"fixtures not found at {FIXTURES_DIR} — run `pnpm fixture`")
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def fixture_traces(fixtures_dir: Path) -> list[dict]:
    traces = []
    for path in sorted(fixtures_dir.glob("trace-*.json")):
        traces.append(json.loads(path.read_text()))
    assert traces, "no fixtures found"
    return traces


@pytest_asyncio.fixture()
async def pool():
    try:
        pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
    except (OSError, asyncpg.PostgresError):
        pytest.skip("postgres not running — docker-compose up -d")
    from app.storage.schema_check import verify_schema

    ok, detail = await verify_schema(pool)
    if not ok:
        await pool.close()
        pytest.skip(f"schema not migrated ({detail}) — pnpm db:migrate")
    yield pool
    await pool.close()
