# Roadmap

Phased delivery from `project-idea.md`, scoped to the MVP first
(spec §45), then V2/V3. Each milestone is independently demoable and ends in
a commit. **Status is updated as milestones complete.**

## MVP — "a real live neural trace" (spec Phases 1–3)

| Milestone | Scope | Exit criterion | Status |
|---|---|---|---|
| **M0 Scaffold** | Monorepo, workspace scripts, Postgres (OrbStack), Drizzle schema + migration, FastAPI `/health`, web shell, git | `pnpm dev` brings up all three services; 3 tables in psql | ✅ done |
| **M1 Contract** | `packages/trace-schema`: zod + types + `SIGNAL_TAXONOMY` + seeded generator + 3 fixtures; Pydantic mirror; contract tests both runtimes | vitest + pytest both green on the *same* fixture files; `pnpm fixture` byte-stable | ✅ done |
| **M2 Design system + landing** | tokens package, fonts, editorial hero `/`, hairline sections, looping fixture teaser, `/methodology` skeleton | Page reads as "AI investigation instrument"; Signal accent used only for attention-worthy change | ✅ done |
| **M3 Explorer on fixtures** | `/explore` fixture mode: TraceCanvas + node registry, Inspector with taxonomy badges, TokenStream, EntropyMeter, ProbabilityDistribution, LayerActivityPanel, ⌘K palette | python-rust fixture plays end-to-end; every number matches the fixture JSON; mobile stacks | ✅ done (interactive E2E in M6) |
| **M4 Live inference** | TransformerLensBackend (BASIC mode), SSE `POST /trace`, device self-check, asyncpg persistence, `mode:'live'` | "Why is the sky blue?" streams with per-token p/entropy/latency; events persisted; replayable | ✅ done — tr_ao1pg9wr: 10 tokens/2.9 s on MPS, self-check pass, 13 events persisted + replayed |
| **M5 Layer activity** | STANDARD mode: resid capture, RunningNormalizer, `LAYER_ACTIVITY` events, live panel, `docs/Interpretability.md` | Layer panel animates from real measurements; entropy matches independent NumPy recompute | ✅ done — entropy agrees to 1e-14 at equal precision (fp32 pipeline: 2e-4 bits); l2Norms match the `resid_post = resid_mid + mlp_out` identity to 1e-5 |
| **M6 Replay + polish** | `/trace/[id]` playback, `/traces`, error/loading states, Playwright E2E | Full §45 checklist green on desktop + 390px | 🔜 next |

## V2 — interpretation & investigation (spec Phases 4–6)

- Attention visualization tied to the token stream (Phase 4) — hook-side
  reduction, never full pattern tensors
- Semantic concept layer, labeled **INTERPRETED** (Phase 5)
- Separate uncertainty quantities (model uncertainty / input ambiguity /
  answer stability) (spec §22)
- Counterfactual mode: "what would change the answer?" (Phase 6)
- Trace search via pgvector similarity (spec §28)
- Model comparison: same prompt through multiple registered models

## V3 — research grade (spec Phase 7)

- Activation patching / interventions
- Sparse autoencoders, feature discovery, attribution graphs
- Downloadable traces, research notebooks

## Deliberately not built (spec §45)

Accounts, social features, teams, billing, model marketplace, fancy RAG,
autonomous agents, mobile app. These distract from the research/product core.
