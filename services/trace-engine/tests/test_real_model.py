"""Real GPT-2 backend smoke test — opt-in: `uv run pytest -m model`.

Downloads ~500 MB on first run (cached afterwards). Verifies what the fake
backend cannot: real tokenizer ids, real log-prob shapes, MPS-vs-CPU
self-check, greedy-continuation stability, and — for STANDARD mode —
independent NumPy recomputation of every derived number we ship
(entropy, layer norms).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

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


async def test_entropy_matches_independent_numpy_recompute(backend: TransformerLensBackend):
    """M5 exit criterion: entropyBits equals an independent recomputation.

    Independent path: naive float64 softmax of the RAW logits (not
    log_softmax) + Shannon's −Σ p·ln p in nats → bits — a different
    softmax and a different entropy formula than entropy_from_log_probs.

    Two comparisons:
      * same precision (float64 log-probs) → agreement to ~1e-9: the
        formulas are mathematically identical
      * shipped fp32 pipeline → 1e-3 bits: the residual gap is float32
        quantization of log_softmax (measured ~1.8e-4 bits), below the
        4-decimal resolution entropyBits is reported at. The engine is
        fp32 end-to-end by design (MPS safety), so this is precision
        policy, not math error.
    """
    import torch  # noqa: F811 — local import keeps module import torch-free

    ctx = backend.encode("The capital of France is")
    with torch.no_grad():
        final = backend.model(torch.tensor([ctx]))[0, -1].float()

    z = final.double().numpy()
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    h_independent = float(
        -(p * np.log(p, where=p > 0, out=np.zeros_like(p))).sum() / math.log(2)
    )

    same_precision = entropy_from_log_probs(
        torch.log_softmax(final.double(), dim=-1).numpy()
    )
    assert abs(same_precision - h_independent) < 1e-9

    res = backend.step(ctx, collect_layers=False)
    shipped = entropy_from_log_probs(res.full_log_probs)  # fp32 pipeline
    assert shipped == pytest.approx(h_independent, abs=1e-3)


async def test_layer_norms_match_residual_stream_identity(backend: TransformerLensBackend):
    """M5 exit criterion: LAYER_ACTIVITY l2Norms are real measurements.

    GPT-2's residual stream satisfies the exact identity
        resid_post[i] = resid_mid[i] + mlp_out[i]
    (attention block output already folded into resid_mid). We rebuild
    resid_post from two INDEPENDENTLY cached tensors, take the NumPy
    l2 norm of the rebuilt vector, and require it to equal the backend's
    reported per-layer l2Norm — plus the identity itself to hold to
    float32 exactness.
    """
    import torch

    ctx = backend.encode("The capital of France is")
    res = backend.step(ctx, collect_layers=True)

    def wanted(name: str) -> bool:
        return name.endswith(("hook_resid_post", "hook_resid_mid", "hook_mlp_out"))

    with torch.no_grad():
        _, cache = backend.model.run_with_cache(
            torch.tensor([ctx]), names_filter=wanted, prepend_bos=False
        )
        try:
            for layer, l2_backend in res.layer_stats:
                i = layer - 1
                rebuilt = cache[f"blocks.{i}.hook_resid_mid"][0, -1] + cache[
                    f"blocks.{i}.hook_mlp_out"
                ][0, -1]
                direct = cache[f"blocks.{i}.hook_resid_post"][0, -1]
                # exact residual-stream identity (same computation graph)
                assert float((rebuilt - direct).abs().max()) < 1e-4
                # independent NumPy norm of the rebuilt vector
                n_rebuilt = float(np.linalg.norm(rebuilt.numpy().astype(np.float64)))
                assert n_rebuilt == pytest.approx(l2_backend, abs=1e-2)
        finally:
            del cache
