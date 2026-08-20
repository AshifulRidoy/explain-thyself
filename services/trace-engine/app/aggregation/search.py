"""Trace search (spec §28) — Python mirror of packages/trace-schema/src/search.ts.

Shared constants and the pgvector text-literal encoding. The embedding
itself comes from the backend's `embed_prompt` (the model's own final-layer
prompt representation); similarity is cosine, computed by Postgres.
"""

from __future__ import annotations

# Column dimension == d_model of the registered model (GPT-2 small 768;
# the fake matches so one column serves both). A future model with a
# different d_model needs its own column — search is per-representation,
# never cross-model.
EMBEDDING_DIM = 768

# One string, both runtimes — the UI never invents its own caption.
SEARCH_BASIS = (
    "cosine similarity of the model's final-layer prompt representation "
    "(mean resid_post over prompt tokens) — rank orders how the model "
    "represents these prompts; it is not semantic meaning, and GPT-2's "
    "hidden space is anisotropic, so absolute values are compressed"
)

_DEFAULT_LIMIT = 10
_MAX_LIMIT = 50


def clamp_limit(limit: int | None) -> int:
    if limit is None:
        return _DEFAULT_LIMIT
    return max(1, min(_MAX_LIMIT, limit))


def vector_literal(embedding: list[float]) -> str:
    """pgvector text input ('[0.123,-0.456,…]') — asyncpg has no native
    vector codec, so the parameter is sent as text and cast $n::vector."""
    return "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
