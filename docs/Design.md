# Design — the instrument's visual language

Near-monochrome, editorial, clinical. The interface should feel like a
scientific instrument that happens to be beautiful — never like a SaaS
dashboard. (Spec §6–8, §32–34.)

## Palette

| Token | Value | Meaning |
|---|---|---|
| Paper | `#F2F1EC` | page ground |
| Ink | `#161616` | text |
| Muted | `#747474` | secondary text, machine labels |
| Line | `#C7C6C0` | hairlines |
| Panel | `#EBEAE5` | recessed surfaces |
| **Signal** | `#E85A3F` | **"something changed / deserves attention" — nothing else** |

The accent is not "AI". It marks: live streaming state, the selected node's
ring, entropy spikes, a value that just changed. If a use of Signal doesn't
mean one of those, it's wrong.

## Typography (a duality — part of the identity)

| Voice | Face | Use |
|---|---|---|
| Editorial | Instrument Serif | identity: "Explain The Self", concept headings — enormous |
| Interface | Inter | controls, body copy |
| Machine | IBM Plex Mono | ALL machine state: `TRACE 0142`, `LAYER 08`, `TOKEN 017`, `0.82` |

Type scale: `--text-xs` 11px (machine labels) → `--text-hero` `clamp(3.5rem,
12vw, 8rem)`. Hierarchy is done by typography, not by boxes.

## Structural rules

1. **Hairlines over containers.** `─────` divides; cards/boxes only where a
   physically distinct object needs one (graph nodes).
2. **No SaaS stat cards.** Machine state is mono label/value rows:
   ```
   MODEL / GPT-2 SMALL
   ────────────────────────
   STATUS             READY
   TOKENS               012
   LAYERS                12
   TRACE                LIVE
   ```
3. **Data feels discovered, not announced.** `CONFIDENCE 0.82` in mono with a
   thin hairline bar — small, clinical, precise. Never a big colorful gauge.
4. Numbers come from the trace; the UI renders `layerCount` from data — never
   hardcodes.

## Epistemic badges (product-critical)

Every measured quantity in the Inspector carries its tier from
`SIGNAL_TAXONOMY`: `MEASURED` / `DERIVED` / `INTERPRETED`. INTERPRETED rows
are visually distinct (muted, italic label) — an interpretation is not a
measurement. `/methodology` renders the full taxonomy with definitions.

## Component inventory (deliberately small — spec §32)

**Shell** `AppShell` (hairline-divided layout) · `SiteHeader` (wordmark ·
Explore/Traces/Methodology · ⌘K) · `CommandPalette`

**Trace** `TraceHeader` (mono status row) · `TraceCanvas` (React Flow v12,
client-only) · `nodeRegistry` (**event `type` → node component — the spec §15
mapping**) · nodes `InputNode · TokenNode · ConceptNode · OutputNode` (+ shared
`TraceNode` chrome) · `Inspector` · `TokenStream` · `TraceControls` · `PlaybackControls`

**Data viz** `LayerActivityPanel` (L01… rows of hairline bars) ·
`EntropyMeter` (current + sparkline, d3-scale/shape only) ·
`ProbabilityDistribution` (top-k rows) · `MiniBars` (12-bar strip — the "live
neural trace" inside a TokenNode)

**System** `ConnectionStatus · TraceStatus · LoadingState · ErrorState`

Later phases add: `AttentionMap`, `ActivationMap`, `ConceptCluster`,
`EvidenceList`, `CounterfactualSlider`, `ModelSelector`.

## Motion (state, not decoration — spec §33)

| What | Duration |
|---|---|
| Token appearance in stream | 100–250 ms |
| Node entrance on canvas | 150–300 ms |
| Panel transitions | 200–400 ms |

Never animate: static text, borders, cards, hovers. The interface feels alive
because data moves, not because CSS moves.

## Layout

Desktop (`/explore`): `graph | inspector` over a full-width token stream.
Mobile (390px): stacked `trace header → graph → selected node → token stream →
inspector`. The three-column desktop layout is never squeezed onto a phone.

## Graph topology (MVP)

`INPUT → token₀ → token₁ → … → OUTPUT` chain with `ConceptNode`s attached at
their positions; deterministic hand-rolled wrap layout (no dagre/elk). A
`TokenNode` carries: token text, probability, entropy spark, 12-layer
MiniBars — the node IS the measurement.
