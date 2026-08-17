
---

# Explain The Self — Complete Build Plan

## 0. The product in one sentence

> **Explain The Self is an interactive microscope for language models, showing how tokens, activations, attention, concepts, evidence, uncertainty, and counterfactual inputs relate to model behavior.**

Your current tagline works very well:

> **A transparent interface for navigating the latent reasoning paths of synthetic intelligence.**

I would keep it.

---

# 1. What you're actually building

There are really **three systems** inside the project.

```text
                    EXPLAIN THE SELF
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     MODEL ENGINE      TRACE ENGINE       UI
          │                │                │
     runs model        transforms        visualizes
     captures          activations       everything
     activations       into signals
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                     USER EXPERIENCE
```

### System A — Model Engine

Runs an open-weight transformer and captures things such as:

* tokens
* logits
* hidden states
* attention
* layer activations
* activation norms
* token probabilities
* selected intermediate activations

Hugging Face Transformers exposes hidden states and attention tensors when requested, while TransformerLens provides a more interpretability-focused activation cache and hooks. ([Hugging Face][1])

### System B — Trace Engine

Turns those enormous tensors into **human-readable signals**.

For example:

```text
RAW ACTIVATION
       ↓
normalization
       ↓
layer aggregation
       ↓
token relevance
       ↓
concept association
       ↓
trace event
```

### System C — Interface

Turns the trace into:

* neural graph
* token stream
* reasoning timeline
* activation map
* uncertainty
* evidence
* counterfactuals
* model state

This is where your existing visual design becomes extremely valuable.

---

# 2. The most important scientific principle

Don't claim:

> **“This is the AI's actual thought process.”**

Instead say:

> **“This interface visualizes measurable internal signals and derived interpretations of model behavior.”**

Why?

Because an activation isn't inherently:

> “The model is thinking about climate.”

You need an interpretation layer between the raw tensor and the human-readable concept.

This is actually where the research becomes interesting.

TransformerLens explicitly focuses on inspecting internal activations and intervening on them, while its `run_with_cache()` API lets you collect selected activations during a forward pass. ([Transformer Lens][2])

So your product can honestly distinguish:

```text
RAW SIGNAL
activation norm
attention pattern
logit probability

        ↓

DERIVED SIGNAL
token relevance
layer activity
activation cluster

        ↓

INTERPRETATION
"historical context"
"economic reasoning"
"uncertainty"
```

That distinction should appear in your UI.

---

# 3. Architecture

I'd build it as a **monorepo with two runtimes**.

The frontend/backend application is TypeScript.

The model/interpretability engine is Python.

```text
                         ┌─────────────────────┐
                         │       Browser       │
                         │                     │
                         │ Next.js             │
                         │ React               │
                         │ React Flow          │
                         │ D3                  │
                         └──────────┬──────────┘
                                    │
                              SSE / WebSocket
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Trace API        │
                         │                     │
                         │ FastAPI             │
                         │ Python              │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                  TransformerLens        Hugging Face
                         │                     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              OPEN MODEL
                                    │
                   ┌────────────────┼───────────────┐
                   ▼                ▼               ▼
              activations       attention         logits
                   │                │               │
                   └────────────────┼───────────────┘
                                    ▼
                              TRACE ENGINE
                                    │
                                    ▼
                               TRACE JSON
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Postgres       │
                         │                     │
                         │ sessions            │
                         │ traces              │
                         │ evaluations         │
                         │ embeddings          │
                         └─────────────────────┘
```

---

# 4. Recommended tech stack

## Frontend

### Next.js

Use the current App Router architecture with TypeScript.

Next.js's App Router is filesystem-based and supports Server Components, Suspense, Server Functions, route handlers, and the modern React architecture. ([Next.js][3])

Use:

```text
Next.js
TypeScript
Tailwind CSS
```

---

## UI primitives

### shadcn/ui + Base UI

As of July 2026, **Base UI is the default primitive base for new shadcn projects**, while Radix remains supported. ([Shadcn UI][4])

Use shadcn for:

* dialogs
* command palette
* tabs
* dropdowns
* tooltips
* buttons
* inputs
* resizable panels
* tables

But **do not let shadcn dictate your visual design**.

You're building your own interface.

---

# 5. Visualization stack

This is probably the most important frontend decision.

## React Flow

Use React Flow for the main trace graph.

Its custom node system allows you to build your own node components and put essentially arbitrary interactive content inside them. ([React Flow][5])

Create:

```text
ConceptNode
TokenNode
ActivationNode
EvidenceNode
DecisionNode
UncertaintyNode
CounterfactualNode
```

For example:

```text
┌──────────────────────────┐
│ CONCEPT                  │
│                          │
│ historical anchoring     │
│                          │
│ activation      0.82     │
│ confidence      0.74     │
└──────────────────────────┘
```

---

## D3

Use D3 only where you need custom mathematical/data visualization.

Good candidates:

* activation heatmaps
* token/layer matrices
* entropy curves
* probability distributions
* sensitivity graphs
* attribution visualizations
* embedding projections

Don't use D3 to build the entire application.

---

## Motion

Use Motion for:

* node appearance
* trace progression
* panel transitions
* token generation
* subtle state changes

Avoid:

> “Everything is animated because AI.”

Your animation should communicate **state**.

---

# 6. Fonts

I'd use:

### Editorial

A high-character serif for:

> Explain
> The Self

Your screenshot's typography is already on the right track.

### Interface

**Inter**

### Machine data

**IBM Plex Mono**

So:

```text
Explain The Self
       ↑
   editorial

TRACE_0142
LAYER 08
TOKEN 017
CONFIDENCE 0.82
       ↑
    machine
```

This duality is part of the identity.

---

# 7. Design system

Your screenshot suggests a very strong visual direction.

Don't abandon it.

## Palette

Use an almost monochrome system.

```text
Paper          #F2F1EC
Ink            #161616
Muted          #747474
Line           #C7C6C0
Panel          #EBEAE5
Signal         #E85A3F
```

You don't need all of these immediately.

In fact, start almost entirely monochrome.

The accent should mean:

> **something changed**

or

> **something deserves attention**

rather than simply:

> “AI.”

---

# 8. Design principles

### Rule 1 — No generic SaaS cards

Avoid:

```text
┌─────────────┐
│ AI MODEL    │
│             │
│ 92%         │
└─────────────┘
```

everywhere.

Instead use:

```text
MODEL / GPT-X
────────────────────────
STATUS             READY
TOKENS              412
LAYERS               24
TRACE               LIVE
```

---

### Rule 2 — Hairlines over containers

Your screenshot already does this well.

Use:

```text
────────────────────────────
```

instead of:

```text
╭──────────────────────────╮
│                          │
│                          │
╰──────────────────────────╯
```

for most structural divisions.

---

### Rule 3 — Typography does the hierarchy

Use:

* enormous serif → identity
* medium serif → concepts
* sans → interface
* mono → machine state

---

### Rule 4 — Data should feel discovered

Don't present:

> **Confidence: 82%**

inside a giant colorful progress bar.

Instead:

```text
CONFIDENCE

0.82

████████████████░░░░
```

or even:

```text
CONFIDENCE  0.82
```

Small, clinical, precise.

---

# 9. The homepage

Your existing screenshot is basically the correct starting point.

I'd structure the landing page:

```text
EXPLAIN
THE SELF

A transparent interface for navigating the
latent reasoning paths of synthetic intelligence.


[ LIVE TRACE VISUALIZATION ]

LIVE NEURAL TRACE // 0.982 CONFIDENCE


────────────────────────────────

TRACING LOGIC

────────────────────────────────

REASONING NODES                         ENTROPY

01  Conceptual Anchoring                0.18
    ...
```

Then eventually:

**Scroll**

↓

### Enter a trace

```text
WHAT WOULD YOU LIKE TO EXAMINE?

[ Ask the model something...              ]
```

↓

### Trace begins

↓

### Interactive investigation

---

# 10. Main application

I would have the actual product at:

```text
/explore
```

The layout:

```text
┌─────────────────────────────────────────────────────────┐
│ EXPLAIN THE SELF                 TRACE 0142     ⌘K      │
├───────────────────────────────────┬─────────────────────┤
│                                   │                     │
│                                   │   INSPECTOR         │
│                                   │                     │
│        TRACE CANVAS               │   selected node     │
│                                   │                     │
│          ●────●                   │   activation        │
│         /      \                  │   evidence          │
│       ●          ●                │   related concepts │
│        \        /                 │                     │
│          ●────●                   │                     │
│                                   │                     │
├───────────────────────────────────┴─────────────────────┤
│ TOKEN STREAM                                             │
│                                                         │
│ The climate impact of nuclear energy ...                │
└─────────────────────────────────────────────────────────┘
```

Three major areas:

### Trace Canvas

Your graph.

### Inspector

Everything about the selected node.

### Token Stream

The generated output, synchronized with the trace.

---

# 11. The Trace Canvas

This is your signature feature.

When a model is processing/generating, nodes should appear.

For example:

```text
USER INPUT
    │
    ▼
[interpretation]
    │
    ├───────────────┐
    ▼               ▼
[climate]       [energy]
    │               │
    └───────┬───────┘
            ▼
        [policy]
            │
            ▼
       [synthesis]
            │
            ▼
          OUTPUT
```

But the underlying nodes can have actual activation metrics.

---

# 12. Your “LIVE NEURAL TRACE”

This is where the project gets technically interesting.

You can use actual model internals.

Hugging Face can expose:

* hidden states
* attention
* logits
* generation scores

during generation. ([Hugging Face][1])

TransformerLens gives you deeper access to cached intermediate activations. Its `run_with_cache()` mechanism can capture activations and allows filtering which activation names are cached. ([Transformer Lens][2])

So create a pipeline:

```text
TOKEN
  ↓
embedding
  ↓
layer 1
  ↓
layer 2
  ↓
layer 3
  ↓
...
  ↓
layer N
  ↓
logits
  ↓
next token
```

At each generation step, collect a **small summary**, not the entire tensor.

---

# 13. Don't send raw tensors to the browser

This is crucial.

Don't do:

```text
GPU
 ↓
4GB tensor
 ↓
JSON
 ↓
browser
```

Instead:

```text
GPU
 ↓
raw activation
 ↓
aggregation
 ↓
normalization
 ↓
trace event
 ↓
small JSON
 ↓
browser
```

For example:

```json
{
  "token": "nuclear",
  "position": 17,
  "layer_activity": [
    0.12,
    0.21,
    0.47,
    0.62,
    0.81
  ],
  "entropy": 0.18,
  "top_tokens": [
    {"token": "energy", "probability": 0.42},
    {"token": "power", "probability": 0.17}
  ]
}
```

That's enough to drive your UI.

---

# 14. Define a Trace Event schema

This should be one of the first things you build.

Something like:

```text
Trace
├── metadata
├── input
├── tokens
├── activations
├── concepts
├── evidence
├── decisions
├── uncertainty
└── output
```

Conceptually:

```json
{
  "trace_id": "0142",
  "model": {
    "name": "your-model",
    "revision": "..."
  },
  "input": {
    "text": "..."
  },
  "tokens": [],
  "events": [],
  "concepts": [],
  "uncertainty": {},
  "output": {}
}
```

This becomes your **internal protocol**.

Once you have this, the frontend doesn't care whether the data came from:

* GPT
* Llama
* Gemma
* a local model
* TransformerLens
* a future interpretability technique

It just consumes `Trace`.

That's a very good architectural boundary.

---

# 15. Trace event types

I'd define these:

```text
INPUT
TOKEN
LAYER_ACTIVITY
ATTENTION
CONCEPT
EVIDENCE
HYPOTHESIS
DECISION
UNCERTAINTY
OUTPUT
```

Example:

```json
{
  "type": "CONCEPT",
  "id": "concept_climate",
  "label": "climate impact",
  "score": 0.91,
  "position": 17
}
```

Then your React Flow renderer knows:

```text
type === "CONCEPT"
        ↓
ConceptNode
```

---

# 16. How to derive the neural graph

Don't try to visualize every neuron.

That's useless.

Instead create **aggregated nodes**.

For example:

```text
RAW

12 layers
×
12 heads
×
768 dimensions
```

becomes:

```text
SEMANTIC GROUPS

climate
energy
economics
policy
history
uncertainty
```

Your graph is therefore an **interpretability layer**, not a literal rendering of the neural network.

That's an important conceptual distinction.

---

# 17. Start with token probabilities

This should be your easiest real-time visualization.

Suppose the model is predicting:

> “The capital of France is ___”

You might get:

```text
TOKEN PREDICTION

Paris       0.94
London      0.02
Berlin      0.01
Rome        0.01
Other       0.02
```

Then your entropy score is derived from the distribution.

You can visualize:

```text
ENTROPY

LOW
│
│ ███
│ █████
│ ███████
│ ██████████
│
HIGH
```

This gives you a legitimate quantitative signal before you've solved difficult concept attribution.

---

# 18. Then add hidden-state activity

For each layer:

```text
L01  0.12
L02  0.19
L03  0.31
L04  0.27
L05  0.64
L06  0.82
...
```

Visualize:

```text
LAYER ACTIVITY

01  ███
02  ████
03  ██████
04  █████
05  ███████████
06  █████████████
07  ███████
```

This becomes your first real **LIVE NEURAL TRACE**.

---

# 19. Then attention

Attention tensors can be enormous: Hugging Face describes them as per-layer, per-head matrices over sequence positions. ([Hugging Face][1])

Don't show every head.

Start with:

```text
HEAD AGGREGATION

token → context relevance

nuclear       ██████████
energy        █████████
climate       ███████
policy        ████
the           █
```

Later let advanced users expand:

```text
Layer 08

Head 01
Head 02
Head 03
...
Head 12
```

---

# 20. Semantic concept layer

This is where you can use another model.

For example:

```text
RAW MODEL SIGNALS
        ↓
embedding / activation representation
        ↓
semantic clustering
        ↓
concept labels
```

You might obtain:

```text
CLUSTER 01

energy
electricity
power
reactor
generation

→ "ENERGY SYSTEMS"
```

Important:

**label this as an interpretation.**

Don't pretend the model literally contains a neuron called:

> `ENERGY SYSTEMS`

unless you've actually established that.

---

# 21. Evidence layer

This is where the project expands beyond pure mechanistic interpretability.

If the model uses retrieval:

```text
QUESTION
   ↓
RETRIEVAL
   ↓
SOURCE 01
SOURCE 02
SOURCE 03
   ↓
ANSWER
```

Show:

```text
EVIDENCE

01  IPCC report
    relevance 0.91

02  research paper
    relevance 0.84

03  government data
    relevance 0.77
```

This lets users distinguish:

> **What the model internally represented**

from:

> **What external information influenced the answer.**

That's extremely useful.

---

# 22. Uncertainty layer

Don't use a fake “AI confidence” number.

Separate different quantities:

```text
MODEL UNCERTAINTY
0.18

EVIDENCE QUALITY
0.82

INPUT AMBIGUITY
0.41

ANSWER STABILITY
0.76
```

Those are different things.

For example:

### Model uncertainty

Derived from the output probability distribution.

### Input ambiguity

Could be estimated by asking an auxiliary model to classify ambiguity or by generating alternate interpretations.

### Evidence quality

Based on your retrieval/source scoring.

### Answer stability

Run controlled perturbations and see whether the answer changes.

This is much more interesting than one giant:

> `98.2% confidence`

---

# 23. Counterfactual mode

This should become one of the major features.

Put a button in the inspector:

> **WHAT WOULD CHANGE THE ANSWER?**

Click it.

The UI creates:

```text
ORIGINAL

"I am a beginner who wants to learn programming."

→ Python
```

Then:

```text
COUNTERFACTUAL

"I am a systems engineer who needs maximum performance."

→ Rust
```

Show:

```text
VARIABLE              IMPACT

experience            █████████
performance           ██████████
ecosystem             ███████
career goal           ████████
learning speed        ██████
```

This is where the product becomes much more than a visualization.

It becomes an **investigation tool**.

---

# 24. Backend choice

## Python

Use Python specifically for:

* PyTorch
* TransformerLens
* Hugging Face
* NumPy
* scientific processing
* interpretability experiments

Use FastAPI to expose the trace engine.

Example conceptual endpoints:

```text
POST /trace
GET  /trace/:id
POST /trace/:id/counterfactual
GET  /trace/:id/events
```

---

# 25. Realtime transport

Use **SSE initially**.

You don't necessarily need WebSockets.

The model produces:

```text
event 1
event 2
event 3
event 4
...
```

Your browser receives them as they happen.

For example:

```text
SERVER

TOKEN → "The"
        ↓
TOKEN → "climate"
        ↓
ACTIVATION → ...
        ↓
TOKEN → "impact"
        ↓
CONCEPT → climate
```

The UI animates the graph accordingly.

If you later need bidirectional control during inference, move to WebSockets.

---

# 26. AI application layer

For ordinary application-level AI features, I'd use **Vercel AI SDK**.

It supports streaming and structured generation, and current AI SDK documentation describes `Output.object()` for typed structured output. ([Vercel][6])

Use it for things like:

* interpreting a trace
* generating concept labels
* summarizing a trace
* creating counterfactual questions
* natural-language explanations

Don't use it as your actual neural instrumentation layer.

Think:

```text
TransformerLens
     ↓
actual model internals

AI SDK
     ↓
interpretation / product AI
```

Different jobs.

---

# 27. Database

Use:

**PostgreSQL + Drizzle ORM**

Drizzle has native PostgreSQL support and supports pgvector for vector similarity search. ([orm.drizzle.team][7])

Your initial tables:

```text
users
sessions
traces
trace_events
concepts
evidence
counterfactuals
evaluations
```

Potential schema:

```text
traces
────────────────────
id
session_id
model
model_revision
input
output
created_at
duration_ms
token_count
```

```text
trace_events
────────────────────
id
trace_id
type
position
layer
payload
created_at
```

```text
concepts
────────────────────
id
trace_id
label
score
embedding
```

---

# 28. Why pgvector?

Eventually you can search previous traces.

Imagine:

> **Find traces similar to this one.**

The system finds:

```text
TRACE 021
"Why do people become successful?"

TRACE 087
"What causes achievement?"

TRACE 104
"Why does money correlate with success?"
```

Then visualize how the model's internal behavior differed.

That could become an entire research feature.

Drizzle's documentation describes pgvector support for exact and approximate nearest-neighbor search and vector columns/indexes. ([orm.drizzle.team][8])

---

# 29. Project structure

I'd structure the repo like this:

```text
explain-the-self/
│
├── apps/
│   │
│   └── web/
│       ├── app/
│       │   ├── page.tsx
│       │   ├── explore/
│       │   │   └── page.tsx
│       │   ├── trace/
│       │   │   └── [id]/
│       │   │       └── page.tsx
│       │   └── api/
│       │
│       ├── components/
│       │   ├── trace/
│       │   ├── inspector/
│       │   ├── visualization/
│       │   ├── ui/
│       │   └── shell/
│       │
│       ├── lib/
│       │   ├── api/
│       │   ├── db/
│       │   └── utils/
│       │
│       └── styles/
│
├── services/
│   │
│   └── trace-engine/
│       ├── app/
│       │   ├── main.py
│       │   ├── routes/
│       │   └── schemas/
│       │
│       ├── models/
│       ├── instrumentation/
│       ├── aggregation/
│       ├── interpretation/
│       └── tests/
│
├── packages/
│   ├── trace-schema/
│   └── design-tokens/
│
├── research/
│   ├── notebooks/
│   ├── experiments/
│   └── papers/
│
└── docs/
    ├── architecture.md
    ├── design.md
    └── interpretability.md
```

Next.js explicitly supports organizing code under `src`, colocating files, route groups, and feature-oriented structures, so you have flexibility here. ([Next.js][9])

---

# 30. Build it in phases

This is extremely important.

**Do not attempt the final system immediately.**

I'd use **7 phases**.

---

# Phase 1 — Build the visual shell

### Goal

Make your screenshot real.

No AI yet.

Build:

* landing page
* typography
* graph canvas
* reasoning node list
* trace header
* inspector
* token stream
* responsive behavior

Use fake data.

For example:

```text
TRACE_0142

01 Conceptual Anchoring       0.82
02 Causal Association         0.71
03 Evidence Retrieval         0.64
04 Synthesis                  0.88
```

### Success criterion

You should be able to show the page to someone and they immediately understand:

> “This is some kind of AI investigation instrument.”

---

# Phase 2 — Real model inference

Introduce a local/open model.

Don't worry about deep interpretability yet.

Capture:

* generated tokens
* token probabilities
* latency
* entropy

Now the UI is responding to a real model.

### Success criterion

User enters:

> “Why is the sky blue?”

and watches:

```text
TOKEN 01
Why

TOKEN 02
is

TOKEN 03
the

...
```

appear live.

---

# Phase 3 — Actual activation trace

Now introduce TransformerLens.

Start capturing:

```text
layer activations
```

not everything.

TransformerLens's cache API supports selecting which activations to cache, which is useful because caching everything can become expensive. ([Transformer Lens][2])

Create:

```text
LayerActivityPanel
```

Show:

```text
L01 ███
L02 █████
L03 ███████
L04 ████
...
```

### Success criterion

The “neural trace” is now based on **actual model measurements**.

---

# Phase 4 — Attention + token visualization

Add:

```text
attention
```

and connect it to your token stream.

Example:

```text
THE CLIMATE IMPACT OF NUCLEAR ENERGY

climate      █████████
impact       ███████
nuclear      ██████████
energy       █████████
```

Then let users click a token.

The graph highlights related attention patterns.

### Success criterion

The token stream and trace graph feel synchronized.

---

# Phase 5 — Semantic concepts

Now add the interpretation layer.

Generate concepts:

```text
CLIMATE
ENERGY
POLICY
ECONOMICS
RISK
```

Attach them to trace regions.

Now the UI can show:

```text
MODEL SIGNAL
      ↓
CLUSTER
      ↓
CONCEPT
```

And label this appropriately:

> **Derived interpretation**

rather than presenting it as literal neural truth.

### Success criterion

A user can ask:

> “What concepts were most active?”

and explore them visually.

---

# Phase 6 — Counterfactuals

This is where I would make the product genuinely special.

Add:

> **WHAT WOULD CHANGE THE ANSWER?**

Run controlled prompt variations.

Compare:

```text
original
vs
counterfactual
```

Then show:

```text
ANSWER STABILITY

███████████░░

changed variables:

performance       +0.42
experience        +0.21
ecosystem         -0.13
```

### Success criterion

The user can manipulate a variable and watch the model's behavior change.

---

# Phase 7 — Research-grade features

Only after everything above works.

Add:

### Activation patching

Intervene on activations and see whether behavior changes.

### Feature visualization

Explore learned features.

### Sparse autoencoders

Potentially map activations into more interpretable feature spaces.

### Attribution graphs

Build causal/attribution-style graphs.

### Cross-model comparison

Run the same prompt through:

```text
MODEL A
MODEL B
MODEL C
```

and compare traces.

This is where your project can move from **portfolio project → serious interpretability research tool**.

---

# 31. The most impressive demo

If this were my project, I would make the launch demo extremely simple.

Prompt:

> **“Should I learn Python or Rust?”**

Then show:

```text
TRACE STARTING
```

The graph wakes up.

Tokens appear:

```text
Should
I
learn
Python
or
Rust
?
```

Then:

```text
INTERPRETATION

career decision
beginner context
language comparison
```

Then concept nodes appear:

```text
Python ──────── ecosystem
   │                │
   │                │
learning ───── career
   │                │
   └──────┬─────────┘
          │
         Rust
```

Then:

```text
CANDIDATE WEIGHTING

Python     ████████████
Rust       ████████
```

Then final:

> **Python is probably the better starting point...**

Then the user clicks:

> **WHAT WOULD CHANGE THE ANSWER?**

The UI reveals:

```text
PERFORMANCE
████████████████

SYSTEMS PROGRAMMING
██████████████

BEGINNER FRIENDLINESS
████████

ECOSYSTEM
██████████████
```

Move:

**Performance → 100**

The trace changes.

> **Rust**

That demo communicates the entire product in ~30 seconds.

---

# 32. Your actual UI component inventory

I'd deliberately limit yourself to a small set.

### Core

```text
AppShell
TraceHeader
TraceCanvas
TraceNode
TraceEdge
Inspector
TokenStream
```

### Data

```text
ActivationMap
EntropyMeter
ProbabilityDistribution
LayerTimeline
AttentionMap
ConceptCluster
EvidenceList
```

### Interaction

```text
CommandPalette
TraceControls
CounterfactualSlider
ModelSelector
PlaybackControls
```

### System

```text
ConnectionStatus
TraceStatus
LoadingState
ErrorState
```

Don't create 100 components.

Make these **extremely good**.

---

# 33. Animation rules

Your animation system should follow the model.

### Generation

Fast:

```text
100–250ms
```

### Node creation

Slight delay:

```text
150–300ms
```

### Panel transition

Slow:

```text
200–400ms
```

### Graph changes

Use interpolation.

### Never animate:

* static text
* every border
* every card
* every hover

The interface should feel **alive because data is moving**, not because CSS is constantly moving.

---

# 34. Mobile

Your screenshot is already mobile-oriented.

Keep it.

But the full explorer should become:

```text
mobile:

TRACE
──────

graph

──────

selected node

──────

token stream

──────

inspector
```

On desktop:

```text
graph | inspector
──────┴──────────
token stream
```

Don't attempt to preserve the desktop three-column interface on a 390px screen.

---

# 35. Performance architecture

This project can get expensive quickly.

Remember that attention and hidden-state tensors can be large, and autoregressive generation already involves substantial computation. Hugging Face's documentation explains that generation proceeds token by token and uses KV caching to avoid recomputing prior attention states. ([Hugging Face][10])

Therefore:

### Never stream raw tensors.

### Never store every tensor forever.

### Don't capture every layer by default.

Instead create modes:

```text
TRACE MODE

BASIC
├─ tokens
├─ entropy
└─ logits

STANDARD
├─ basic
├─ hidden states
└─ attention summary

RESEARCH
├─ all above
├─ selected activations
└─ interventions
```

This is also a nice product feature.

---

# 36. The “Research Mode”

I'd put a tiny control somewhere:

```text
MODE

● OBSERVE
○ ANALYZE
○ RESEARCH
```

### Observe

Beautiful, simple.

```text
concepts
tokens
confidence
```

### Analyze

More information.

```text
layers
attention
activation
uncertainty
```

### Research

Everything.

```text
hooks
heads
features
interventions
raw metrics
```

This prevents your beautiful interface from becoming a nightmare for normal users.

---

# 37. Testing strategy

You need three categories.

## Frontend

Use:

**Vitest**

for logic.

**Playwright**

for:

* graph interaction
* trace playback
* counterfactual flow
* responsive layout

---

## Trace engine

Unit-test:

```text
activation aggregation
entropy
normalization
token mapping
trace serialization
```

---

## Scientific validation

This is the most important.

For every visualization ask:

> **What exactly does this number mean?**

For example:

```text
ENTROPY = Shannon entropy of normalized next-token distribution
```

Good.

But:

```text
CONCEPT = 0.82
```

is meaningless unless you define how it was obtained.

Create a methodology page.

---

# 38. Build a `/methodology` page

This could become one of the strongest parts of the website.

Explain:

### What is measured?

> Hidden states, attention distributions, token probabilities, etc.

### What is derived?

> Aggregated activation scores, semantic clusters, stability metrics.

### What is inferred?

> Human-readable interpretations of those signals.

Use three labels:

```text
MEASURED
DERIVED
INTERPRETED
```

This is extremely aligned with the project's philosophy of transparency.

---

# 39. Example methodology UI

```text
SIGNAL TAXONOMY

MEASURED
──────────────
Hidden state
Attention
Logits
Token probability

DERIVED
──────────────
Activation magnitude
Entropy
Similarity
Stability

INTERPRETED
──────────────
Concept
Intent
Reasoning stage
Semantic label
```

That's a very powerful design element.

---

# 40. Don't call everything “reasoning”

This is another important distinction.

Use terms like:

**Trace**

**Activation**

**Representation**

**Prediction**

**Attention**

**Evidence**

**Interpretation**

**Decision**

rather than constantly saying:

> “The AI thought X.”

Your product will sound significantly more scientifically mature.

---

# 41. What the final navigation could look like

I'd keep it tiny:

```text
EXPLAIN THE SELF

Explore
Traces
Models
Methodology
About
```

Maybe:

```text
⌘K
```

for everything else.

Don't build a giant dashboard navigation.

---

# 42. Deployment

For the web application:

**Vercel**

is a natural choice with Next.js.

For the Python trace engine:

Initially:

```text
Docker
+
GPU server
```

Don't try to force your model inference into a serverless function.

Eventually:

```text
Vercel
   │
   ▼
Trace API
   │
   ▼
GPU inference server
```

Your frontend should never know where the model is running.

---

# 43. Local development

Your development environment could look like:

```text
localhost:3000
    ↓
Next.js

localhost:8000
    ↓
FastAPI

localhost:5432
    ↓
Postgres

GPU
    ↓
TransformerLens
```

Then a request:

```text
Browser
  ↓
Next.js
  ↓
FastAPI
  ↓
model
  ↓
Trace events
  ↓
SSE
  ↓
browser
```

---

# 44. The development order I'd personally follow

If you're building this yourself, **don't start with the model**.

Do:

### Week 1

Design system.

### Week 2

Landing page.

### Week 3

Trace explorer with fake data.

### Week 4

Real model inference.

### Week 5

Token probabilities + entropy.

### Week 6

Hidden-state visualization.

### Week 7

Attention.

### Week 8

Semantic concepts.

### Week 9

Counterfactuals.

### Week 10

Polish + methodology + demo.

The exact timing can vary substantially, but the dependency order matters.

---

# 45. MVP definition

Your MVP should be only:

```text
✓ Ask question
✓ Model generates answer
✓ Tokens stream live
✓ Token probabilities
✓ Entropy
✓ Layer activity
✓ Interactive graph
✓ Click a node
✓ Inspector
✓ Save trace
✓ Replay trace
```

**That's enough.**

Don't initially build:

```text
✗ accounts
✗ social features
✗ teams
✗ billing
✗ giant model marketplace
✗ dozens of models
✗ fancy RAG
✗ autonomous agents
✗ mobile app
```

Those will distract you from the actual research/product idea.

---

# 46. V2

Then:

```text
V2

✓ semantic concepts
✓ attention visualization
✓ evidence
✓ counterfactuals
✓ trace comparison
✓ model comparison
✓ trace search
```

---

# 47. V3

Then:

```text
V3 / Research

✓ activation patching
✓ feature discovery
✓ sparse autoencoders
✓ attribution graphs
✓ intervention experiments
✓ cross-model analysis
✓ downloadable traces
✓ research notebooks
```

At this point, you have something that could plausibly become a serious open-source interpretability project.

---

# 48. Your final stack, locked in

If you want me to make the decision for you, **this is the stack I'd use**:

```text
┌───────────────────────────────────────────┐
│                 FRONTEND                  │
├───────────────────────────────────────────┤
│ Next.js                                   │
│ TypeScript                                │
│ Tailwind CSS                              │
│ shadcn/ui + Base UI                       │
│ Lucide                                    │
│ Motion                                    │
│ React Flow                                │
│ D3                                        │
└───────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│                  AI APP                   │
├───────────────────────────────────────────┤
│ Vercel AI SDK                             │
│ Structured outputs                        │
│ Streaming                                 │
└───────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│              INTERPRETABILITY             │
├───────────────────────────────────────────┤
│ Python                                    │
│ PyTorch                                   │
│ Hugging Face Transformers                 │
│ TransformerLens                           │
│ NumPy                                     │
└───────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│                 BACKEND                   │
├───────────────────────────────────────────┤
│ FastAPI                                   │
│ SSE initially                             │
│ WebSockets later if necessary             │
└───────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│                  DATA                     │
├───────────────────────────────────────────┤
│ PostgreSQL                                │
│ Drizzle ORM                               │
│ pgvector                                  │
└───────────────────────────────────────────┘

┌───────────────────────────────────────────┐
│              INFRASTRUCTURE               │
├───────────────────────────────────────────┤
│ Vercel                                    │
│ Docker                                    │
│ GPU inference server                      │
│ Sentry                                    │
│ PostHog                                   │
└───────────────────────────────────────────┘
```

The frontend stack is particularly well suited to your concept: React Flow supports custom interactive nodes, Next.js gives you the application shell/API structure, and Vercel's AI SDK handles streaming and structured AI output. ([React Flow][5])

---

# 49. The most important architectural decision

If you remember **one thing from this entire guide**, make it this:

### Don't make your frontend understand neural networks.

Make it understand **Trace**.

```text
                    MODEL
                      │
          ┌───────────┴───────────┐
          │                       │
       OpenAI                Local model
          │                       │
          │                  TransformerLens
          │                       │
          └───────────┬───────────┘
                      ▼
                 TRACE ENGINE
                      │
                      ▼
                 TRACE SCHEMA
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Browser      Database    Research
```

Then you can change the underlying model without rebuilding the product.

And your interface becomes a **general-purpose language-model observability layer**.

---

# 50. And this is what I'd ultimately make the product promise

Not:

> **“See inside an AI's brain.”**

That's gimmicky.

Instead:

> **Explain The Self**
>
> **A microscope for artificial intelligence.**
>
> Explore what a model sees, what it represents, what it predicts, what evidence influences it, and what changes its behavior.

And the UI can literally communicate the epistemic hierarchy:

```text
┌──────────────────────────────────────────┐
│              EXPLAIN THE SELF             │
│                                          │
│              TRACE 0142                  │
├──────────────────────────────────────────┤
│                                          │
│  MEASURED                                │
│  ────────                                │
│  activations                             │
│  attention                               │
│  logits                                  │
│                                          │
│  DERIVED                                 │
│  ───────                                 │
│  entropy                                 │
│  relevance                               │
│  stability                               │
│                                          │
│  INTERPRETED                             │
│  ───────────                             │
│  concepts                                │
│  intent                                  │
│  reasoning stages                        │
│                                          │
└──────────────────────────────────────────┘
```
