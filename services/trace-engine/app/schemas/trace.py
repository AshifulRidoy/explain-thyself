"""Pydantic mirror of packages/trace-schema (the Trace contract).

1:1 with the TypeScript/Zod side. Both runtimes validate the SAME committed
fixture files in their test suites — that is the contract test. extra="forbid"
everywhere so field drift fails loudly here instead of silently in the browser.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------- primitives

EpistemicLevel = Literal["MEASURED", "DERIVED", "INTERPRETED"]

TraceEventType = Literal[
    "INPUT",
    "TOKEN",
    "LAYER_ACTIVITY",
    "ATTENTION",
    "CONCEPT",
    "EVIDENCE",
    "HYPOTHESIS",
    "DECISION",
    "UNCERTAINTY",
    "OUTPUT",
]

TraceMode = Literal["BASIC", "STANDARD", "RESEARCH"]
TraceStatus = Literal["streaming", "complete", "error"]

Probability = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TopToken(StrictModel):
    tokenId: int = Field(ge=0)
    text: str
    rawText: str
    leadingSpace: bool
    probability: Probability
    rank: int = Field(ge=0)


class InputToken(StrictModel):
    position: int = Field(ge=0)
    tokenId: int = Field(ge=0)
    text: str


# ------------------------------------------------------------------- events

class TraceEventBase(StrictModel):
    id: str = Field(pattern=r"^evt_\d{4}$")
    seq: int = Field(ge=0)
    type: TraceEventType
    t: int = Field(ge=0)


class InputEvent(TraceEventBase):
    type: Literal["INPUT"] = "INPUT"
    level: Literal["MEASURED"] = "MEASURED"
    text: str
    tokenCount: int = Field(ge=0)
    tokens: list[InputToken] = Field(max_length=512)


class TokenEvent(TraceEventBase):
    type: Literal["TOKEN"] = "TOKEN"
    level: Literal["MEASURED"] = "MEASURED"
    position: int = Field(ge=0)
    step: int = Field(ge=0)
    tokenId: int = Field(ge=0)
    text: str
    rawText: str
    leadingSpace: bool
    probability: Probability
    rank: int = Field(ge=0)
    entropyBits: float = Field(ge=0, le=20)
    topK: list[TopToken] = Field(min_length=1, max_length=8)
    latencyMs: float = Field(ge=0)


class LayerStat(StrictModel):
    layer: int = Field(ge=0)
    l2Norm: float = Field(gt=0)
    normRatio: float = Field(gt=0)


class LayerActivityEvent(TraceEventBase):
    type: Literal["LAYER_ACTIVITY"] = "LAYER_ACTIVITY"
    level: Literal["DERIVED"] = "DERIVED"
    position: int = Field(ge=0)
    step: int = Field(ge=0)
    layers: list[LayerStat] = Field(min_length=1)


class AttentionAggregate(StrictModel):
    # -1 = the prepended BOS token (GPT-2's attention sink) — a real
    # measured position, not a visible stream token
    position: int = Field(ge=-1)
    text: str
    weight: Probability


class AttentionEvent(TraceEventBase):
    type: Literal["ATTENTION"] = "ATTENTION"
    level: Literal["DERIVED"] = "DERIVED"
    position: int = Field(ge=0)
    layer: int = Field(ge=0)
    aggregated: Optional[list[AttentionAggregate]] = None
    headEntropyBits: Optional[list[float]] = None


class ConceptEvidence(StrictModel):
    tokenId: int = Field(ge=0)
    text: str
    probability: Probability


class ConceptEvent(TraceEventBase):
    type: Literal["CONCEPT"] = "CONCEPT"
    level: Literal["INTERPRETED"] = "INTERPRETED"
    # concept identity (e.g. "concept_parsing") — distinct from the event id
    conceptId: str
    label: str
    # probability mass on the dictionary word set — the MEASURED half;
    # the label is the INTERPRETED half
    score: Probability
    positions: Optional[list[int]] = None
    # the measured tokens carrying the mass, so the label stays auditable
    evidence: Optional[list[ConceptEvidence]] = None


class EvidenceEvent(TraceEventBase):
    type: Literal["EVIDENCE"] = "EVIDENCE"
    level: Literal["MEASURED"] = "MEASURED"
    evidenceId: str
    label: str
    source: str
    relevance: Probability


class HypothesisEvent(TraceEventBase):
    type: Literal["HYPOTHESIS"] = "HYPOTHESIS"
    level: Literal["INTERPRETED"] = "INTERPRETED"
    hypothesisId: str
    text: str
    confidence: Probability


class DecisionEvent(TraceEventBase):
    type: Literal["DECISION"] = "DECISION"
    level: Literal["DERIVED"] = "DERIVED"
    decision: Literal["sampled", "greedy", "stop", "max_tokens", "aborted"]
    detail: Optional[str] = None


class UncertaintyWindow(StrictModel):
    fromStep: int = Field(ge=0)
    toStep: int = Field(ge=0)


class StabilityVariant(StrictModel):
    perturbation: str
    text: str
    agreedTokens: int = Field(ge=0)
    totalTokens: int = Field(ge=0)
    divergedPositions: list[int] = Field(default_factory=list)


class UncertaintyEvent(TraceEventBase):
    type: Literal["UNCERTAINTY"] = "UNCERTAINTY"
    # None when the quantity was considered and deliberately NOT measured —
    # then value is None too and basis carries the reason
    level: Optional[Literal["MEASURED", "DERIVED"]] = None
    kind: Literal[
        "MODEL_UNCERTAINTY", "EVIDENCE_QUALITY", "INPUT_AMBIGUITY", "ANSWER_STABILITY"
    ]
    value: Optional[Probability] = None
    # always present: the method for a measured value, the reason for a null
    basis: str
    window: Optional[UncertaintyWindow] = None
    # ANSWER_STABILITY evidence: one row per perturbation actually rerun
    variants: Optional[list[StabilityVariant]] = None


class OutputEvent(TraceEventBase):
    type: Literal["OUTPUT"] = "OUTPUT"
    level: Literal["MEASURED"] = "MEASURED"
    text: str
    tokenCount: int = Field(ge=0)
    durationMs: float = Field(ge=0)
    finishReason: str


TraceEvent = Annotated[
    Union[
        InputEvent,
        TokenEvent,
        LayerActivityEvent,
        AttentionEvent,
        ConceptEvent,
        EvidenceEvent,
        HypothesisEvent,
        DecisionEvent,
        UncertaintyEvent,
        OutputEvent,
    ],
    Field(discriminator="type"),
]


# ------------------------------------------------------------------ envelope


class TraceModelInfo(StrictModel):
    name: str
    revision: str
    device: str
    layerCount: int = Field(gt=0)
    paramCount: int = Field(gt=0)


class TraceSampling(StrictModel):
    maxTokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    topK: Optional[int] = None
    seed: Optional[int] = None


class TraceOutput(StrictModel):
    text: str
    tokenCount: int = Field(ge=0)
    durationMs: float = Field(ge=0)
    finishReason: str


class InputText(StrictModel):
    text: str


class Trace(StrictModel):
    id: str = Field(pattern=r"^tr_[0-9a-z]{2,12}$")
    displayId: int = Field(gt=0)
    model: TraceModelInfo
    input: InputText
    traceMode: TraceMode
    sampling: TraceSampling
    status: TraceStatus
    createdAt: str
    output: Optional[TraceOutput] = None
    events: list[TraceEvent] = []


# ----------------------------------------------------- counterfactuals (V2)

# API artifacts, not TraceEvents: the original trace stays immutable; a
# counterfactual comparison persists in its own `counterfactuals` table.
# Mirrors counterfactualResultSchema / counterfactualRequestSchema in
# packages/trace-schema/src/counterfactuals.ts.


class CounterfactualResult(StrictModel):
    id: str = Field(pattern=r"^cf_[0-9a-z]{6,12}$")
    traceId: str = Field(pattern=r"^tr_")
    # INTERPRETED variable label, or CUSTOM_VARIABLE ("your edit")
    variable: str = Field(min_length=1)
    # null for free-form edits (no single word was manipulated)
    originalWord: Optional[str] = Field(default=None, min_length=1)
    replacementWord: Optional[str] = Field(default=None, min_length=1)
    promptText: str = Field(min_length=1)
    outputText: str
    # compared length = the ORIGINAL trace's emitted token count
    tokenCount: int = Field(gt=0)
    agreedTokens: int = Field(ge=0)
    # 1 − agreed/tokenCount; 0 = the answer survived the edit byte-identical
    impact: float = Field(ge=0, le=1)
    # first step where the answers differ; None when identical
    firstDivergence: Optional[int] = Field(default=None, ge=0)
    # signed mean entropy shift vs. the original answer (bits/token)
    entropyDelta: float
    basis: str = Field(min_length=1)
    createdAt: str = Field(min_length=1)


class CounterfactualRequest(StrictModel):
    # all — run every applicable dictionary substitution (capped);
    # one — rerun a specific resolved variable; prompt — free-form edit
    scope: Literal["all", "one", "prompt"]
    variable: Optional[str] = None
    originalWord: Optional[str] = None
    prompt: Optional[str] = Field(default=None, min_length=1, max_length=2000)


# ------------------------------------------------------------- search (V2)

# Spec §28: "find traces similar to this one." The embedding is the
# model's own final-layer prompt representation; similarity is cosine in
# Postgres via pgvector. Mirrors searchHitSchema / searchResponseSchema
# in packages/trace-schema/src/search.ts (SEARCH_BASIS lives in
# app/aggregation/search.py and is echoed verbatim into responses).


class SearchHit(StrictModel):
    traceId: str = Field(pattern=r"^tr_")
    displayId: int = Field(gt=0)
    input: str = Field(min_length=1)
    # cosine ∈ [-1, 1]; the RANK is the signal, not the absolute value
    similarity: float = Field(ge=-1, le=1)
    modelName: str = Field(min_length=1)
    traceMode: TraceMode
    tokenCount: Optional[int] = Field(default=None, ge=0)
    createdAt: str = Field(min_length=1)


class SearchResponse(StrictModel):
    # the free-text query, or the source trace's prompt for /similar
    query: str
    basis: str = Field(min_length=1)
    # already ranked (similarity desc); empty is a valid answer
    results: list[SearchHit] = []
    # how many stored traces carry an embedding and were compared
    searchable: int = Field(ge=0)


class BackfillReport(StrictModel):
    """POST /search/backfill — re-embed traces recorded before the
    embedding column existed (embedding depends only on prompt text and
    model, both stored, so this is a re-derivation, not a guess)."""
    filled: int = Field(ge=0)
    remaining: int = Field(ge=0)


# ------------------------------------------------------ comparison (V2)
#
# Spec Phase 7, pulled into V2 by the Roadmap: the same prompt through
# two REGISTERED models, compared. Model B's run is a full persisted
# trace; the comparison is a separate artifact attached to trace A (the
# counterfactuals immutability rule). Token agreement is only defined
# when the two models share a tokenizer — the route rejects the request
# otherwise rather than compute a number about nothing.
# Mirrors comparisonResultSchema / comparisonRequestSchema in
# packages/trace-schema/src/comparison.ts.


class ComparisonResult(StrictModel):
    id: str = Field(pattern=r"^cmp_[0-9a-z]{6,12}$")
    # the anchor trace — the one this comparison hangs off
    traceIdA: str = Field(pattern=r"^tr_")
    # the freshly recorded run through model B
    traceIdB: str = Field(pattern=r"^tr_")
    modelA: str = Field(min_length=1)
    modelB: str = Field(min_length=1)
    # the prompt both models answered
    prompt: str = Field(min_length=1)
    # each answer's own emitted token count
    tokenCountA: int = Field(gt=0)
    tokenCountB: int = Field(gt=0)
    # min(tokenCountA, tokenCountB) — the positions compared
    comparedLength: int = Field(gt=0)
    agreedTokens: int = Field(ge=0)
    # agreed/comparedLength; 1 = identical over the compared range
    agreement: float = Field(ge=0, le=1)
    # first compared position where the token ids differ; None when identical
    firstDivergence: Optional[int] = Field(default=None, ge=0)
    outputTextA: str
    outputTextB: str
    # mean shipped entropyBits of each answer (bits/token)
    meanEntropyA: float = Field(ge=0)
    meanEntropyB: float = Field(ge=0)
    # meanB − meanA, signed
    entropyDelta: float
    basis: str = Field(min_length=1)
    createdAt: str = Field(min_length=1)


class ComparisonRequest(StrictModel):
    # registry key of the model to run the prompt through (model B)
    model: str = Field(min_length=1)
