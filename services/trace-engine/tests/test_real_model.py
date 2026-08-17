"""Real GPT-2 backend smoke test — opt-in: `uv run pytest -m model`.

Downloads ~500 MB on first run (cached afterwards). Verifies what the fake
backend cannot: real tokenizer ids, real log-prob shapes, MPS-vs-CPU
self-check, and that greedy GPT-2 continuation is stable across runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.aggregation.stats import entropy_from_log_probs
from app.models.registry import MODEL_REGISTRY
from app.models.transformer_lens_backend import TransformerLensBackend

pytestmark = [pytest.mark.model, pytest.mark.asyncio]


@pytest.fixture(scope="module")
def backend():
    spec = MODEL_REGISTRY["gpt2-small"]
    b = TransformerLensBackend(spec)
    b.load("cpu")  # CPU ground truth — the MPS caveat (#1178)
    return b


async def test_matches_hf_reference_numerics(backend: TransformerLensBackend):
    """Regression anchor, verified against plain HuggingFace gpt2 (BOS
    prepended, fp32): top-1 is ' now' at p≈0.0475. Base GPT-2 is genuinely
    uncertain on this prompt — if this anchor moves, numerics broke."""
    ctx = backend.encode("The capital of France is")
    r = backend.step(ctx, collect_layers=False)
    assert r.top_k[0].raw_text == "Ġnow"
    assert r.top_k[0].probability == pytest.approx(0.0475, abs=1e-3)


async def test_greedy_continuation_is_stable(backend: TransformerLensBackend):
    ctx = backend.encode("The capital of France is")
    r1 = backend.step(ctx, collect_layers=False)
    r2 = backend.step(ctx, collect_layers=False)
    assert r1.top_k[0].token_id == r2.top_k[0].token_id
    assert abs(r1.top_k[0].probability - r2.top_k[0].probability) < 1e-6


async def test_entropy_and_layer_stats_shapes(backend: TransformerLensBackend):
    ctx = backend.encode("Why is the sky blue?")
    r = backend.step(ctx, collect_layers=True)
    assert r.full_log_probs.shape == (50257,)
    assert float(np.exp(r.full_log_probs.max())) > 0.01
    entropy = entropy_from_log_probs(r.full_log_probs)
    assert 0.0 < entropy < 15.617  # bounded by log2(vocab)
    assert len(r.layer_stats) == 12
    # residual norms grow with depth in GPT-2 — a real, known property
    norms = [l2 for _, l2 in r.layer_stats]
    assert norms[-1] > norms[0]


async def test_prompt_roundtrip_matches_text(backend: TransformerLensBackend):
    toks = backend.prompt_tokens("Why is the sky blue?")
    assert len(toks) == 6
    # composing from rawText + leadingSpace must reproduce the prompt —
    # the UI renders spacing from these flags, never by decoding prefixes
    parts = []
    for tid, _text in toks:
        tok = backend.decode_token(tid)
        prefix = " " if tok.leading_space and not tok.text.startswith((" ", "\n")) else ""
        parts.append(prefix + tok.text)
    assert "".join(parts) == "Why is the sky blue?"
