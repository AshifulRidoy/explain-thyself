# Explain The Self

> A microscope for artificial intelligence. Explore what a model sees, what it
> represents, what it predicts — and what changes its behavior.

An interactive instrument for examining **measurable internal signals** of a
language model (GPT-2 small via TransformerLens) and how they relate to its
generated output. Every signal is labeled by what it epistemically is:
**MEASURED**, **DERIVED**, or **INTERPRETED** — this is an observability
layer, not a claim about "the AI's thoughts".

## Documentation

- [`docs/Architecture.md`](docs/Architecture.md) — the three systems, the Trace contract, data flow
- [`docs/Roadmap.md`](docs/Roadmap.md) — milestones, status, V2/V3
- [`docs/Plan.md`](docs/Plan.md) — execution details: contracts, layout, verification
- [`docs/Design.md`](docs/Design.md) — visual language, component inventory, motion rules
- [`project-idea.md`](project-idea.md) — the original product/spec document

## Quickstart

```bash
pnpm install
cp .env.example apps/web/.env.local   # defaults work for local dev
pnpm dev                              # postgres + migrations + web(:3000) + engine(:8000)
```

First run downloads GPT-2 small (~500 MB) when the first live trace starts.

## Try it

```bash
# stream a trace from the engine directly
curl -N localhost:8000/trace -H 'content-type: application/json' \
  -d '{"prompt":"Why is the sky blue?","maxTokens":20,"traceMode":"STANDARD"}'

curl -s localhost:8000/health | jq
curl -s localhost:8000/trace/tr_XXXXXXXX | jq '.events | length'
```

Or open `http://localhost:3000/explore?fixture=python-rust` for a
deterministic offline demo (no model download).

## Tests

```bash
pnpm test                             # vitest across TS packages
pnpm --filter @ets/trace-engine test  # pytest (engine, incl. contract tests)
```

## Stack

Next.js (App Router, TS, Tailwind v4) · React Flow · D3 · Motion · FastAPI ·
TransformerLens · PyTorch · PostgreSQL + Drizzle · docker-compose (OrbStack)
