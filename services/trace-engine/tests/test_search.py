"""Trace search primitives (spec §28): the embedding seam is deterministic
and normalized, the pgvector literal is well-formed, and the fake's
stand-in vectors don't secretly correlate (which would fake a ranking).
"""

from __future__ import annotations

import math
import re

from app.aggregation.search import SEARCH_BASIS, clamp_limit, vector_literal
from app.models.fake_backend import FakeBackend
from app.models.registry import MODEL_REGISTRY


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_fake_embedding_deterministic_normalized_768() -> None:
    backend = FakeBackend(MODEL_REGISTRY["fake"])
    v1 = backend.embed_prompt("Why is the sky blue?")
    v2 = backend.embed_prompt("Why is the sky blue?")
    assert v1 == v2  # same text → the same vector, always
    assert len(v1) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, abs_tol=1e-4)


def test_fake_embedding_distinguishes_texts_without_shared_bias() -> None:
    """mulberry32 streams from nearby seeds share a constant-direction
    bias (uncentered cosines ≈ 0.76 between ANY two texts); the centering
    in embed_prompt must remove it, or every trace would rank as equally
    'similar' to every other and the fake demo would lie."""
    backend = FakeBackend(MODEL_REGISTRY["fake"])
    sky = backend.embed_prompt("Why is the sky blue?")
    tensors = backend.embed_prompt("Tell me about tensors.")
    python = backend.embed_prompt("Should I learn Python or Rust?")
    # unrelated texts near-orthogonal; identical texts exactly aligned
    assert abs(_cos(sky, tensors)) < 0.2
    assert abs(_cos(tensors, python)) < 0.2
    assert _cos(sky, backend.embed_prompt("Why is the sky blue?")) > 0.999


def test_vector_literal_is_valid_pgvector_input() -> None:
    lit = vector_literal([0.5, -0.25, 0.0])
    assert lit == "[0.500000,-0.250000,0.000000]"
    assert re.fullmatch(r"\[-?\d+\.\d+(,-?\d+\.\d+)*\]", lit)


def test_clamp_limit() -> None:
    assert clamp_limit(None) == 10
    assert clamp_limit(3) == 3
    assert clamp_limit(0) == 1  # a zero/negative ask still returns something
    assert clamp_limit(99) == 50


def test_basis_disclaims_meaning() -> None:
    # the one string both runtimes ship — it must keep saying what the
    # number is NOT (semantic meaning) and why (anisotropy)
    assert "not semantic meaning" in SEARCH_BASIS
    assert "anisotropic" in SEARCH_BASIS
