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
