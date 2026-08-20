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


async def _real_trace_dict(backend, prompt: str, max_tokens: int) -> dict:
    """A complete trace dict from the real pipeline (envelope + events)."""
    raw = ""
    async for frame in trace_stream(
        prompt=prompt,
        trace_mode="STANDARD",
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=None,
        seed=None,
        backend=backend,
        writer=None,
    ):
        raw += frame
    frames = [json.loads(l[6:]) for l in raw.split("\n") if l.startswith("data: ")]
    envelope = frames[0]
    envelope["events"] = frames[1:-1]
    envelope["status"] = "complete"
    return envelope


async def test_counterfactuals_measured_on_real_model(backend: TransformerLensBackend):
    """Counterfactuals (spec §23) on real GPT-2: dictionary variables resolve,
    the unedited CONTROL reproduces the answer exactly (greedy determinism —
    what makes impact a measurement of the edit, not of noise), the math is
    exact, and at least one surface edit changes the real answer."""
    from app.aggregation.counterfactuals import applicable_substitutions
    from app.engine.counterfactual import run_counterfactual

    prompt = "Why is the sky blue?"
    trace = await _real_trace_dict(backend, prompt, max_tokens=10)
    tokens = [e for e in trace["events"] if e.get("type") == "TOKEN"]
    assert len(tokens) == 10

    variables = applicable_substitutions(prompt)
    assert [v["originalWord"] for v in variables] == ["Why", "sky", "blue"]

    # the control: rerunning the UNEDITED prompt agrees on every token
    control = await run_counterfactual(
        trace, prompt_text=prompt, variable="control",
        original_word=None, replacement_word=None, backend=backend,
    )
    assert control.agreedTokens == control.tokenCount == 10
    assert control.impact == 0.0 and control.firstDivergence is None
    # the control re-derives the SAME distributions, so the mean entropy
    # shift is zero to shipped precision (4dp)
    assert control.entropyDelta == 0.0

    results = []
    for v in variables:
        r = await run_counterfactual(
            trace, prompt_text=v["promptText"], variable=v["variable"],
            original_word=v["originalWord"], replacement_word=v["replacementWord"],
            backend=backend,
        )
        assert r.tokenCount == 10
        assert r.impact == pytest.approx(round(1 - r.agreedTokens / 10, 4))
        results.append(r)

    assert any(r.firstDivergence is not None for r in results), (
        "expected at least one edit to change the real answer: "
        + ", ".join(f"{r.originalWord}→{r.replacementWord}={r.impact}" for r in results)
    )

    # determinism: the same edit, run twice, measures identically
    v0 = variables[0]
    again = await run_counterfactual(
        trace, prompt_text=v0["promptText"], variable=v0["variable"],
        original_word=v0["originalWord"], replacement_word=v0["replacementWord"],
        backend=backend,
    )
    assert (again.impact, again.agreedTokens, again.firstDivergence, again.outputText) == (
        results[0].impact, results[0].agreedTokens, results[0].firstDivergence, results[0].outputText
    )


def test_prompt_embedding_is_a_real_representation(backend: TransformerLensBackend):
    """Spec §28 search runs on these vectors — pin what makes ranking
    honest on the real model: deterministic, unit-norm, d_model wide, and
    paraphrase ranks above unrelated for the instrument's canonical
    prompt families.

    Honest counterexample (measured, deliberately NOT asserted): the spec's
    own 'Why do people become successful?' vs 'What causes achievement?'
    scores BELOW an unrelated sky-blue prompt on GPT-2-small's final-layer
    mean — mean-pooled resid_post is a crude representation, which is
    exactly why SEARCH_BASIS disclaims semantic meaning."""
    from app.aggregation.search import EMBEDDING_DIM

    def cos(a: str, b: str) -> float:
        va, vb = backend.embed_prompt(a), backend.embed_prompt(b)
        return sum(x * y for x, y in zip(va, vb))

    v1 = backend.embed_prompt("Why is the sky blue?")
    v2 = backend.embed_prompt("Why is the sky blue?")
    assert v1 == v2  # deterministic: same prompt, same vector, always
    assert len(v1) == EMBEDDING_DIM == backend.spec.d_model
    assert sum(x * x for x in v1) == pytest.approx(1.0, abs=1e-4)

    families = [
        ("Why is the sky blue?", "Why does the sky look blue?", "Should I learn Python or Rust?"),
        ("The capital of France is", "France's capital city is", "Why is the sky blue?"),
        ("Should I learn Python or Rust?", "Is Python or Rust better to learn?", "The capital of France is"),
    ]
    for original, paraphrase, unrelated in families:
        para = cos(original, paraphrase)
        other = cos(original, unrelated)
        assert para > other, (original, paraphrase, para, other)


@pytest.fixture(scope="module")
def distil() -> TransformerLensBackend:
    """The second registered model — the distilled 82M GPT-2. Same
    tokenizer and d_model as gpt2-small: the two properties that make
    cross-model token agreement measurable and the embedding column
    valid."""
    spec = MODEL_REGISTRY["distilgpt2"]
    b = TransformerLensBackend(spec)
    b.load("cpu")
    return b


async def test_cross_model_comparison_measured(backend, distil):
    """Spec Phase 7 (V2 cut) on real weights: the same prompt through
    gpt2-small and distilgpt2.

    Pinned:
    * the two models really share token ids (same prompt → same INPUT
      encoding) — the precondition the whole artifact rests on
    * the CONTROL (gpt2-small vs itself) agrees exactly: greedy
      determinism, so any disagreement in the real pair is the models
      differing, not noise
    * the real pair's numbers are well-formed and deterministic on rerun
    * distilgpt2's traces embed (d_model 768), so search keeps working
      per-model with no special casing
    """
    from app.engine.comparison import _split_frame, derive_comparison

    prompt = "Why is the sky blue?"
    trace_a = await _real_trace_dict(backend, prompt, max_tokens=10)

    async def collect(b, max_tokens: int = 10) -> dict:
        run = {"id": "", "ids": [], "entropies": [], "output": ""}
        async for frame in trace_stream(
            prompt=prompt, trace_mode="STANDARD", max_tokens=max_tokens,
            temperature=0.0, top_k=None, seed=None, backend=b, writer=None,
        ):
            event, data = _split_frame(frame)
            if event == "trace":
                run["id"] = json.loads(data)["id"]
            elif event == "trace_event":
                payload = json.loads(data)
                if payload.get("type") == "TOKEN":
                    run["ids"].append(payload["tokenId"])
                    run["entropies"].append(float(payload["entropyBits"]))
                elif payload.get("type") == "OUTPUT":
                    run["output"] = payload.get("text", "")
        return run

    # the precondition, measured not assumed
    assert backend.encode(prompt) == distil.encode(prompt)
    assert backend.spec.d_model == distil.spec.d_model == 768

    # the control: same model twice agrees everywhere
    control_run = await collect(backend)
    control = derive_comparison(
        trace_a, model_b="gpt2-small", trace_b_id=control_run["id"],
        ids_b=control_run["ids"], entropy_b=control_run["entropies"],
        output_b=control_run["output"],
    )
    assert control.agreement == 1.0 and control.firstDivergence is None
    assert control.entropyDelta == 0.0

    # the real pair: two different models, one prompt
    run_b = await collect(distil)
    result = derive_comparison(
        trace_a, model_b="distilgpt2", trace_b_id=run_b["id"],
        ids_b=run_b["ids"], entropy_b=run_b["entropies"],
        output_b=run_b["output"],
    )
    assert result.modelA == "gpt2-small" and result.modelB == "distilgpt2"
    assert 0.0 <= result.agreement <= 1.0
    assert result.comparedLength == min(result.tokenCountA, result.tokenCountB)
    assert result.agreement == pytest.approx(
        round(result.agreedTokens / result.comparedLength, 4)
    )
    if result.firstDivergence is not None:
        assert 0 <= result.firstDivergence < result.comparedLength
        # divergence means at least one disagreeing position
        assert result.agreedTokens < result.comparedLength
    assert result.meanEntropyA >= 0 and result.meanEntropyB >= 0
    assert result.entropyDelta == pytest.approx(
        round(result.meanEntropyB - result.meanEntropyA, 4), abs=1e-4
    )

    # determinism: the pair measures identically on rerun
    again = await collect(distil)
    rerun = derive_comparison(
        trace_a, model_b="distilgpt2", trace_b_id=again["id"],
        ids_b=again["ids"], entropy_b=again["entropies"],
        output_b=again["output"],
    )
    assert (rerun.agreement, rerun.agreedTokens, rerun.firstDivergence,
            rerun.outputTextB, rerun.entropyDelta) == (
        result.agreement, result.agreedTokens, result.firstDivergence,
        result.outputTextB, result.entropyDelta)

    # search stays per-model: distilgpt2 embeds into the same 768-dim
    # column and its vectors live in their own representation space
    from app.aggregation.search import EMBEDDING_DIM

    v = distil.embed_prompt(prompt)
    assert len(v) == EMBEDDING_DIM == 768
