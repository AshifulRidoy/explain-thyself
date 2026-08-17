# Architecture

> Explain The Self is an interactive microscope for language models: it shows
> measurable internal signals (tokens, logits, hidden-state activations) and
> how they relate to generated behavior.

## The one governing rule

**The frontend never understands neural networks. It understands `Trace`.**

```
                MODEL (gpt2-small today, swap via registry)
                          │
                 TransformerLens  /  FakeBackend
                          │
                  TRACE ENGINE (Python, FastAPI)
                  aggregation → small JSON events
                          │
                    TRACE SCHEMA (the contract)
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
     Browser            Postgres          Research
   (Next.js, SSE)     (save/replay)      (later)
```

Everything below exists to keep `packages/trace-schema` the only coupling
point between the engine and the interface. Swap the model, swap the
interpretability technique — the product does not change.

## The three systems

| System | Where | Job |
|---|---|---|
| **Model Engine** | `services/trace-engine/app/models/` | Runs the model, captures logits + per-layer residual activations per generation step |
| **Trace Engine** | `services/trace-engine/app/{engine,aggregation}/` | Reduces tensors to small honest scalars; emits `TraceEvent`s over SSE |
| **Interface** | `apps/web` | Consumes `Trace`: graph, token stream, inspector, data viz |

## Monorepo layout

```
apps/web/                  Next.js App Router (TS, Tailwind v4, React Flow, D3, Motion)
services/trace-engine/     FastAPI + TransformerLens (uv-managed Python 3.11 venv)
packages/trace-schema/     THE contract: zod + Pydantic mirror + fixtures + generator
packages/design-tokens/    palette / type / spacing as CSS custom properties + TS consts
docs/                      this documentation
```

Root scripts (`pnpm dev`) orchestrate Postgres (docker-compose via OrbStack),
migrations, web (:3000) and engine (:8000) together.

## Data flow for one trace

1. Browser `POST :8000/trace` **directly** (CORS; Next proxies nothing — SSE
   must not pass through the dev proxy).
2. Engine runs the generation loop: one `asyncio.to_thread` forward pass per
   step; `run_with_cache` captures only `blocks.{i}.hook_resid_post`; only the
   final position is read; `cache.clear()` after every step.
3. Per step the engine emits tiny events (`TOKEN`, `LAYER_ACTIVITY`) — never
   tensors. Entropy is Shannon (log2) over the **full** vocab in log-softmax
   form; layer activity is ‖resid‖₂ + a running-mean ratio.
4. Events stream to the browser (`event: trace` → `trace_event`* → one
   terminal `done`|`error`) and are batch-inserted to Postgres via an async
   queue (DB latency can never delay an SSE frame).
5. `/trace/[id]` replays the stored events through the identical UI component
   tree (`payload` jsonb is the exact validated event JSON).

## Persistence ownership (hard rule)

**Drizzle (TypeScript) is the sole DDL author.** Python writes via asyncpg
raw SQL and verifies the schema at startup (`information_schema`) instead of
creating anything. Tables: `traces`, `trace_events`, `concepts`.

## Model swapping

`app/models/registry.py` holds `ModelSpec` records. Registering Qwen2.5-0.5B
or Gemma later is one line + a smoke test. The `fake` backend (torch-free,
deterministic mulberry32 stream — bit-identical port of the TS fixture
generator) stays forever as the test double and offline demo mode
(`ETS_MODEL=fake`).

## Device strategy (Apple Silicon)

TransformerLens has reported silently-wrong GPT-2 outputs on MPS
(TransformerLensOrg/TransformerLens#1178). Therefore: always float32, and on
first load a **numerics self-check** compares a fixed calibration prompt on
the chosen device vs CPU (top-1 token + max |Δlogit| ≤ 0.5); mismatch ⇒ loud
warning + automatic CPU fallback, surfaced in `GET /health.selfCheck`.

## Known accepted costs (MVP)

- No KV-cache reuse across steps (`run_with_cache` re-runs the prefix each
  step). Fine at ≤256 context × 124M params (tens of ms/step). First
  optimization target after MVP.
- No attention capture (Phase 4). When added: compute head-mean reductions
  *inside* `run_with_hooks`; never cache `[heads, seq, seq]` patterns.

## Ports

web `:3000` · engine `:8000` · postgres `:5432`
