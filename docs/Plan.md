# Plan — execution details

Companion to `Roadmap.md` (what + status). This file holds **how**: the
contracts, file layout, and verification recipe for each milestone. Source of
truth for where anything lives.

## The Trace contract (M1 — frozen)

`packages/trace-schema` is canonical (TS + zod); `app/schemas/trace.py` is the
Pydantic mirror. Both validate the committed fixtures in `packages/trace-schema/fixtures/`
on every test run — that is the contract test. `extra: forbid` / `.strict()`
everywhere; drift fails loudly.

Event types: `INPUT · TOKEN · LAYER_ACTIVITY · ATTENTION · CONCEPT · EVIDENCE ·
HYPOTHESIS · DECISION · UNCERTAINTY · OUTPUT`. Envelope: `Trace` (id, displayId,
model, input, traceMode, sampling, status, createdAt, output?, events[]).

Key decisions baked in:
- **Entropy lives on `TOKEN`** (not a separate per-token event) — halves event
  volume; epistemic honesty is carried by `SIGNAL_TAXONOMY` badges instead.
- `ATTENTION|CONCEPT|EVIDENCE|HYPOTHESIS|UNCERTAINTY` are defined now, emitted
  in later phases (fixtures emit `CONCEPT` so `ConceptNode` is proven day one).
- `seq` is gapless and totally orders a trace; node keys, DB ordering and
  replay scrubbing all key off it.
- Fixtures are byte-stable (mulberry32; fixed createdAt). The Python
  `FakeBackend` ports the same PRNG bit-for-bit (pinned value:
  `mulberry32(42)() == 0.6011037519201636` on both sides).

## Engine internals (M4/M5)

```
app/main.py                 FastAPI, CORS (browser → :8000 direct), lifespan pool
app/config.py               ETS_* settings
app/routes/traces.py        POST /trace (SSE), GET /trace/{id}[,/events]
app/routes/meta.py          GET /health, /models
app/models/registry.py      ModelSpec table (swap = one line)
app/models/backend.py       ModelBackend Protocol: load/encode/prompt_tokens/
                            decode_token/step/is_eos
app/models/{transformer_lens,fake}_backend.py
app/engine/generate.py      THE single generation loop (BASIC + STANDARD)
app/engine/sse.py           format_sse / heartbeat
app/aggregation/stats.py    entropy (log-softmax form), top-k, Welford
                            RunningNormalizer, greedy/temperature sampling
app/storage/*               db pool, batched writer, reader, schema check
```

SSE stream shape: `event: trace` → `event: trace_event`* → exactly one
terminal `event: done|error`. Greedy decoding by default (temperature 0) so
replays and Playwright assertions are deterministic.

## Web app (M2/M3/M6)

Routes: `/` (editorial hero + fixture teaser + examine input) · `/explore`
(the instrument; `?fixture=` | `?prompt=`) · `/trace/[id]` (playback) ·
`/traces` (list) · `/methodology` (signal taxonomy page).

**The swap seam** — `apps/web/lib/trace/useTraceDataSource.ts`,
`mode ∈ {fixture, live, replay}`, all feeding the same pure `applyEvent`
reducer → zustand store. Phase 1→2 is changing one string. Pure and
unit-tested: `applyEvent`, `graph.ts` (state → React Flow nodes/edges),
`parseSse` (chunk-boundary-safe fetch-stream parser — `EventSource` cannot
POST a JSON body, hence fetch-streaming).

Component inventory lives in `Design.md`.

## Verification recipe

- `pnpm test` — vitest (schema determinism, realism invariants, reducer,
  graph, SSE parser) across all TS packages
- `pnpm --filter @ets/trace-engine test` — pytest (entropy anchors: delta→0,
  uniform over 50257→15.617 bits, hand-computed 4-symbol; aggregation; SSE
  framing; full fake stream shape; DB round trip when marked `db`)
- `curl -N localhost:8000/trace -H 'content-type: application/json' -d '{"prompt":"Why is the sky blue?","maxTokens":20,"traceMode":"STANDARD"}'`
- Playwright (M6): fixture explore → live trace → saved playback, desktop + 390px

## Rules of engagement

1. Never send tensors to the browser; never cache an ActivationCache across steps.
2. Drizzle owns DDL; Python verifies, never creates.
3. Every UI-visible number has a `SIGNAL_TAXONOMY` entry with a definition.
4. The Signal accent means "something changed / deserves attention" — nothing else.
5. Fixtures and the FakeBackend stay deterministic forever (they are the test
   doubles; deleting them breaks the contract tests).
