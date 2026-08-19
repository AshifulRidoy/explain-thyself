"""Real GPT-2 backend smoke test — opt-in: `uv run pytest -m model`.

Downloads ~500 MB on first run (cached afterwards). Verifies what the fake
backend cannot: real tokenizer ids, real log-prob shapes, MPS-vs-CPU
self-check, greedy-continuation stability, and — for STANDARD mode —
independent NumPy recomputation of every derived number we ship
(entropy, layer norms).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from app.aggregation.concepts import CONCEPT_DICTIONARY, ConceptScorer
from app.aggregation.stats import entropy_from_log_probs
from app.engine.generate import trace_stream
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


async def test_attention_reduced_hook_side_matches_full_cache(backend: TransformerLensBackend):
    """Phase 4 exit criterion: the hook-side reduction is exactly the
    head-mean final query row of the REAL pattern tensor, computed
    independently by caching the full [heads, q, k] block the forbidden
    way (fine in a test, never in the engine) and reducing in NumPy."""
    import torch

    ctx = backend.encode("The capital of France is")
    res = backend.step(ctx, collect_layers=False, collect_attention=True)

    with torch.no_grad():
        _, cache = backend.model.run_with_cache(
            torch.tensor([ctx]),
            names_filter=lambda n: n.endswith("attn.hook_pattern"),
            prepend_bos=False,
        )
        try:
            for layer in range(12):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, :, -1, :].float()
                independent = pattern.mean(dim=0).numpy().astype(np.float64)
                shipped = res.attention[layer].astype(np.float64)
                assert np.abs(shipped - independent).max() < 1e-6
                # softmax over keys: the row is a distribution
                assert shipped.sum() == pytest.approx(1.0, abs=1e-5)
        finally:
            del cache


async def test_attention_head_entropies_and_bos_sink(backend: TransformerLensBackend):
    """Per-head entropy is bounded by log2(context); GPT-2's attention
    sink means BOS mass grows with depth (a real, known property — if it
    inverts, the layer indexing is wrong)."""
    import math

    ctx = backend.encode("Why is the sky blue?")
    res = backend.step(ctx, collect_layers=False, collect_attention=True)

    for layer in range(12):
        row = res.attention[layer]
        assert len(row) == len(ctx)  # attends exactly over the context
        for h in res.head_entropies[layer]:
            assert 0.0 <= h <= math.log2(len(ctx)) + 1e-6

    bos = [float(res.attention[i][0]) for i in range(12)]
    assert bos[-1] > bos[0]


async def test_every_dictionary_word_resolves_to_real_tokens(backend: TransformerLensBackend):
    """Phase 5 exit criterion: the dictionary is scoreable at all.

    Every word must be a single token in the leading-space form (" word")
    — the form that realizes the word in running text, which is where
    next-token mass lives. A word failing this would silently score 0
    forever. Words that are ALSO single-token bare (~2/3 of the
    dictionary; GPT-2 merges are space-prefixed) additionally score the
    bare id via the resolver.
    """
    tokenizer = backend.model.tokenizer
    bare_single = 0
    total = 0
    for concept in CONCEPT_DICTIONARY:
        for word in concept.words:
            total += 1
            spaced = tokenizer.encode(f" {word}", add_special_tokens=False)
            assert len(spaced) == 1, f"{word!r} is not a single spaced token"
            ids = backend.word_token_ids(word)
            assert spaced[0] in ids and len(ids) <= 2
            if len(tokenizer.encode(word, add_special_tokens=False)) == 1:
                bare_single += 1
    # tokenizer sanity floor — a mass migration to multi-token bare forms
    # would mean the tokenizer or dictionary drifted
    assert bare_single >= total * 0.5


async def test_concept_mass_matches_independent_softmax(backend: TransformerLensBackend):
    """Phase 5 exit criterion: CONCEPT scores are exact probability mass.

    Independent path: plain forward → naive float64 softmax of the raw
    logits → Σ p over each concept's ids. Must equal ConceptScorer's mass
    (computed from the shipped fp32 log-softmax) to float32 exactness.
    Also: the shipped array is a true log-softmax, and every evidence row
    is a member of the concept's own word set carrying its exact share.
    """
    ctx = backend.encode("Why is the sky blue?")
    res = backend.step(ctx, collect_layers=False)

    total_prob = float(np.exp(res.full_log_probs.astype(np.float64)).sum())
    assert total_prob == pytest.approx(1.0, abs=1e-4)

    with torch.no_grad():
        logits = backend.model(torch.tensor([ctx]))[0, -1].float()
    z = logits.double().numpy()
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()

    scorer = ConceptScorer(backend.word_token_ids, backend.token_text)
    for scored in scorer.score(res.full_log_probs):
        ids = sorted({i for w in scored.spec.words for i in backend.word_token_ids(w)})
        assert scored.mass == pytest.approx(float(p[ids].sum()), abs=1e-6)
        assert 0.0 <= scored.mass <= 1.0
        probs = [prob for _, _, prob in scored.evidence]
        assert probs == sorted(probs, reverse=True)
        for token_id, text, prob in scored.evidence:
            assert token_id in ids
            assert text in scored.spec.words  # auditable: a dictionary word
            assert prob == pytest.approx(float(p[token_id]), abs=1e-6)


async def test_sky_blue_prompt_measures_science_concept(backend: TransformerLensBackend):
    """The headline Phase 5 claim, on the real model: while GPT-2 answers
    "Why is the sky blue?", the science/nature word set carries measurable
    probability mass (greedy ⇒ deterministic). The mass is measured; the
    label is the interpretation."""
    raw = ""
    async for frame in trace_stream(
        prompt="Why is the sky blue?",
        trace_mode="STANDARD",
        max_tokens=10,
        temperature=0.0,
        top_k=None,
        seed=None,
        backend=backend,
        writer=None,
    ):
        raw += frame
    events = [
        json.loads(line[6:]) for line in raw.split("\n") if line.startswith("data: ")
    ]
    concepts = [e for e in events if e.get("type") == "CONCEPT"]
    assert concepts, "a real trace should light up at least one concept"

    words = {
        w for c in CONCEPT_DICTIONARY if c.conceptId == "concept_science" for w in c.words
    }
    science = [e for e in concepts if e["conceptId"] == "concept_science"]
    assert science, "the sky-blue answer must light up science/nature"
    for event in science:
        assert event["score"] >= 0.05
        assert event["evidence"]
        for evidence in event["evidence"]:
            assert evidence["text"] in words


async def test_research_emits_verified_uncertainty_layer(backend: TransformerLensBackend):
    """The uncertainty layer on the real model (spec §22): model uncertainty
    must be recomputable from the trace's own TOKEN events against the real
    vocab (50257), answer stability must be a measured agreement over real
    greedy reruns of perturbed prompts, and the two unmeasurable quantities
    must ship as nulls with reasons — never faked."""
    raw = ""
    async for frame in trace_stream(
        prompt="The capital of France is",
        trace_mode="RESEARCH",
        max_tokens=6,
        temperature=0.0,
        top_k=None,
        seed=None,
        backend=backend,
        writer=None,
    ):
        raw += frame
    events = [
        json.loads(line[6:]) for line in raw.split("\n") if line.startswith("data: ")
    ]
    uncertainty = [e for e in events if e.get("type") == "UNCERTAINTY"]
    assert [e["kind"] for e in uncertainty] == [
        "MODEL_UNCERTAINTY",
        "EVIDENCE_QUALITY",
        "INPUT_AMBIGUITY",
        "ANSWER_STABILITY",
    ]

    tokens = [e for e in events if e.get("type") == "TOKEN"]
    model = uncertainty[0]
    assert model["level"] == "DERIVED"
    assert model["basis"].startswith("mean normalized entropy H/log2(50257)")
    mean_norm = sum(t["entropyBits"] for t in tokens) / len(tokens) / math.log2(50257)
    assert model["value"] == pytest.approx(mean_norm, abs=1e-4)

    for skipped in uncertainty[1:3]:
        assert skipped["value"] is None and skipped["level"] is None

    stability = uncertainty[3]
    assert stability["level"] == "MEASURED"
    assert 0 <= stability["value"] <= 1
    variants = stability["variants"]
    assert variants, "lowercase_first always applies to this prompt"
    # the perturbation is real: rerun text differs, token budget matches
    for variant in variants:
        assert variant["text"] != "The capital of France is"
        assert variant["totalTokens"] == len(tokens)
        assert variant["agreedTokens"] == len(tokens) - len(variant["divergedPositions"])
    # greedy determinism makes agreement a property of the perturbation,
    # not sampling noise — a lowercase_first flip is a measured fact
    agreed = sum(v["agreedTokens"] for v in variants)
    assert stability["value"] == pytest.approx(
        round(agreed / (len(variants) * len(tokens)), 4), abs=1e-9
    )


async def test_stability_control_rerun_agrees_exactly(backend: TransformerLensBackend):
    """Control for the stability measurement itself: an UNPERTURBED greedy
    rerun must agree 1.0 — any disagreement there would mean the pipeline
    measures nondeterminism, not sensitivity to the prompt."""
    prompt = "The capital of France is"
    emitted = []
    ctx = backend.encode(prompt)
    for _ in range(5):
        res = backend.step(ctx, collect_layers=False)
        emitted.append(res.top_k[0].token_id)
        ctx.append(emitted[-1])

    # feed the identical prompt through the variant machinery: no perturbation
    # applies to a second identical encode, so we call the comparison directly
    vctx = backend.encode(prompt)
    agreed = 0
    for i, original in enumerate(emitted):
        res = backend.step(vctx, collect_layers=False)
        if res.top_k[0].token_id == original:
            agreed += 1
        vctx.append(res.top_k[0].token_id)
    assert agreed == len(emitted)


async def test_research_path_logits_match_anchor(backend: TransformerLensBackend):
    """The hooks path (RESEARCH) must produce identical logits to the
    plain path — anchored to the HF reference."""
    ctx = backend.encode("The capital of France is")
    hooked = backend.step(ctx, collect_layers=True, collect_attention=True)
    assert hooked.top_k[0].raw_text == "Ġnow"
    assert hooked.top_k[0].probability == pytest.approx(0.0475, abs=1e-3)
