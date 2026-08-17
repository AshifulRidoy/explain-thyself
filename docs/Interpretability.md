# Interpretability

> What we measure inside the model, what we deliberately *don't*, and how
> every number the interface shows is verified. Companion to
> [Architecture.md](Architecture.md) and `/methodology` (the user-facing
> version of the epistemic tiers).

## The stance

We instrument the model, we do not psychoanalyze it. Every signal in a
trace is one of three things (spec §38–39, `SIGNAL_TAXONOMY`):

| Tier | Meaning | Examples |
|---|---|---|
| **MEASURED** | Read directly off a computation | `probability` (softmax of logits), `topK`, `latencyMs`, prompt token ids |
| **DERIVED** | Deterministic math on measurements | `entropyBits` (Shannon, full vocab), `l2Norm` (‖resid‖₂), `normRatio`, `DECISION` |
| **INTERPRETED** | A human-readable claim layered on measurements | concept labels (fixtures only today; Phase 5) |

The interface never hides which tier a number belongs to — the Inspector
badges every field, and "the AI's actual thoughts" is a sentence that
appears nowhere in this codebase.

## What STANDARD mode measures

One hook family, one position, one scalar per layer:

```
names_filter = ^blocks\.(\d+)\.hook_resid_post$      (12 hooks, gpt2-small)
read         = cache[hook][0, -1]                    (final position ONLY)
emit         = ‖·‖₂  →  LAYER_ACTIVITY.l2Norm        (DERIVED)
```

Why this and not something richer:

- **The residual stream is the model's ledger.** Every layer reads and
  writes `resid`; its norm is the honest, architecture-level answer to
  "how much representation has accumulated by layer *i* at this token".
  In GPT-2 it grows roughly monotonically with depth (see below) — a
  real, known property, which is exactly what a demo of "real
  measurements" should show.
- **One scalar per layer keeps the contract small.** 12 floats per step,
  not a 768-vector — `TraceEvent`s stay JSON-cheap, the browser stays
  tensor-free (Architecture.md's one governing rule).
- **BASIC caches nothing.** `traceMode` is a dial, not a flag day: BASIC
  = logits only; STANDARD adds the 12 resid hooks. RESEARCH (later
  phases) will widen the filter — never remove it.

### Why norms grow with depth

Each block *adds* to the stream (`resid_post[i] = resid_mid[i] +
mlp_out[i]`, an exact identity — see verification below) while layer norm
at the *input* of each block keeps what it reads bounded. Additions
accumulate; growth is the expected signature, and traces show it (e.g.
L01 ≈ 60 → L12 ≈ 450 on "The capital of France is"). When a layer's
contribution spikes against its own running baseline, `normRatio` moves —
that is the "something changed" moment the Signal accent exists for.

## normRatio — honest streaming normalization

Raw l2Norms span ~60–450+ across layers; a naive global min-max would
need the *finished* trace and would lie during streaming. Instead each
layer index keeps a **Welford running mean** (`RunningNormalizer`,
`app/aggregation/stats.py`):

```
mean_n = mean_{n-1} + (x_n − mean_{n-1}) / n        # numerically stable
normRatio = x_n / mean_n                             # mean includes x_n
```

Consequences, all intentional:

- first step is **exactly 1.0** by construction (mean of one sample)
- the number means "this step vs. this layer's own recent average at
  this depth" — no cross-layer comparison baked in, no global state
- it streams: every event is final the moment it is emitted
- resetting is per-trace; replays recompute nothing (the ratio ships in
  the event, so replayed traces show exactly what live showed)

## Entropy

`entropyBits` on each TOKEN event is Shannon entropy over the **full
50,257-token vocabulary** — never top-8 truncated (the topK list is
separate and explicitly partial). Computed as `−Σ exp(lp)·lp / ln 2` on
`log_softmax` output (log-sum form: no `p·log p` underflow).

## Verification — every derived number recomputed independently

Both checks run against real weights (`uv run pytest -m model`,
`tests/test_real_model.py`):

1. **Entropy** vs. a path that shares no code: naive float64 softmax of
   the *raw* logits + `−Σ p·ln p`. Agreement: **~1e-14 bits** at equal
   precision (the formulas are identical); the shipped fp32 pipeline
   differs by **~2e-4 bits** (0.002%) — float32 quantization of
   `log_softmax`, below the 4-decimal resolution `entropyBits` is
   reported at. The engine is fp32 end-to-end by design (bf16/f64 are
   unsupported or unsafe on MPS); that is precision policy, not math
   error, and the test pins both facts separately.
2. **Layer norms** vs. the exact residual-stream identity
   `resid_post[i] = resid_mid[i] + mlp_out[i]`: rebuild the vector from
   two *independently cached* hooks, take its NumPy ‖·‖₂, require
   equality with the backend's `l2Norm` (measured max|Δ| ≈ 1e-5; the
   identity itself holds to <1e-4). This proves the norms are read from
   the real residual stream, not an artifact of one hook.

Plus the standing guards: startup device self-check (MPS vs. CPU
top-1 + max|Δlogit|, auto-fallback; TransformerLens#1178), the HF
reference-numerics anchor ("The capital of France is" → " now"
p≈0.0475), and greedy determinism across runs.

## Memory discipline

A retained `ActivationCache` is an OOM waiting to happen, so:

- `names_filter` always narrows the cache (12 tensors, not ~200)
- only the final position `[0, -1]` is ever read
- the cache reference is dropped every step (`del cache` — TL 3.7.2
  caches are plain objects; there is no `.clear()`)
- extraction happens inside `step()`; no tensor leaves the backend
  (`StepResult` carries only scalars + the log-prob vector)

## Not in the MVP — and why

| Signal | Phase | Why deferred |
|---|---|---|
| **Attention patterns** | 4 | `[heads, seq, seq]` is the biggest tensor in the model; must be reduced *hook-side* (head-mean at the final position) before it can enter the contract, never cached raw |
| **Concept labels** | 5 | INTERPRETED tier — requires a labeling method we can stand behind, not a decorative cosine similarity |
| **Uncertainty decomposition** | V2 | model uncertainty vs. input ambiguity vs. answer stability are separate quantities (spec §22) |
| **Counterfactuals / patching** | 6, V3 | interventions change the run itself; needs its own trace semantics |
| **SAE features** | V3 | feature discovery is research-grade; attribution graphs come after |

The pattern to keep: **reduce on the torch side, ship scalars, label the
tier.** If a future signal cannot be honestly named MEASURED or DERIVED,
it must earn the INTERPRETED badge explicitly — or stay out of the trace.
