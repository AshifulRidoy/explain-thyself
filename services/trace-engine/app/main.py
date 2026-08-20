"""Explain The Self — trace engine (FastAPI).

The browser talks to :8000 directly (CORS); Next.js proxies nothing, so SSE
never passes through the dev proxy.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import meta, search, traces

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("ets")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .storage.db import close_pool, init_pool

    pool = await init_pool()
    if pool is None:
        log.warning(
            "serving WITHOUT persistence — POST /trace?persist=true will 503 "
            "until `docker compose up -d && pnpm db:migrate`"
        )
    yield
    await close_pool()


app = FastAPI(title="Explain The Self — Trace Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta.router)
app.include_router(search.router)
app.include_router(traces.router)
