/**
 * The Trace contract (spec §14–15).
 *
 * This is the ONLY coupling point between the trace engine (Python) and the
 * interface (TypeScript). The frontend never sees a tensor — it sees Trace.
 *
 * Every signal carries an epistemic level (spec §2, §38):
 *   MEASURED    — a direct property of the forward pass
 *   DERIVED     — a reduction/statistic computed from measured quantities
 *   INTERPRETED — a human-readable label attached to a signal; not a
 *                 property of any neuron
 */

export type EpistemicLevel = "MEASURED" | "DERIVED" | "INTERPRETED";

export type TraceEventType =
  | "INPUT"
  | "TOKEN"
  | "LAYER_ACTIVITY"
  | "ATTENTION"
  | "CONCEPT"
  | "EVIDENCE"
  | "HYPOTHESIS"
  | "DECISION"
  | "UNCERTAINTY"
  | "OUTPUT";

export type TraceMode = "BASIC" | "STANDARD" | "RESEARCH";

export type TraceStatus = "streaming" | "complete" | "error";

/** A candidate next token from the model's output distribution. */
export interface TopToken {
  tokenId: number;
  /** Display form: BPE space marker stripped. */
  text: string;
  /** BPE form as stored in the tokenizer, e.g. "Ġenergy" (debug/methodology). */
  rawText: string;
  /** True when the BPE form carries a leading space marker (Ġ/Ċ). */
  leadingSpace: boolean;
  probability: number;
  /** 0-based rank in the full distribution. */
  rank: number;
}

/** Shared envelope for every event. `seq` is gapless and totally orders a trace. */
export interface TraceEventBase {
  /** Deterministic: `evt_${seq padded to 4}`. */
  id: string;
  seq: number;
  type: TraceEventType;
  /** Ms since trace start (engine clock) — drives replay pacing. */
  t: number;
}

export interface InputEvent extends TraceEventBase {
  type: "INPUT";
  level: "MEASURED";
  text: string;
  tokenCount: number;
  tokens: { position: number; tokenId: number; text: string }[];
}

export interface TokenEvent extends TraceEventBase {
  type: "TOKEN";
  level: "MEASURED";
  /** Absolute position in the full sequence (prompt length + step). */
  position: number;
  /** 0-based generation step. */
  step: number;
  tokenId: number;
  text: string;
  rawText: string;
  leadingSpace: boolean;
  probability: number;
  rank: number;
  /** Shannon entropy (bits, log2) of the FULL next-token distribution. */
  entropyBits: number;
  /** Top-k candidates; k = 8. Index 0 is the emitted token under greedy decoding. */
  topK: TopToken[];
  /** Wall-clock forward pass for this step. */
  latencyMs: number;
}

export interface LayerStat {
  layer: number;
  /** L2 norm of the residual stream (block output) at the final position. */
  l2Norm: number;
  /** l2Norm / running mean of this layer's l2Norm over steps so far (first step = 1). */
  normRatio: number;
}

export interface LayerActivityEvent extends TraceEventBase {
  type: "LAYER_ACTIVITY";
  level: "DERIVED";
  position: number;
  step: number;
  layers: LayerStat[];
}

export interface AttentionEvent extends TraceEventBase {
  type: "ATTENTION";
  level: "DERIVED";
  position: number;
  layer: number;
  /** Head-mean attention FROM the final position TO each prior position. */
  aggregated?: { position: number; text: string; weight: number }[];
  headEntropyBits?: number[];
}

export interface ConceptEvent extends TraceEventBase {
  type: "CONCEPT";
  level: "INTERPRETED";
  /** Concept identity (e.g. "concept_parsing") — distinct from the event id. */
  conceptId: string;
  label: string;
  score: number;
  positions?: number[];
}

export interface EvidenceEvent extends TraceEventBase {
  type: "EVIDENCE";
  level: "MEASURED";
  /** Evidence identity — distinct from the event id. */
  evidenceId: string;
  label: string;
  source: string;
  relevance: number;
}

export interface HypothesisEvent extends TraceEventBase {
  type: "HYPOTHESIS";
  level: "INTERPRETED";
  /** Hypothesis identity — distinct from the event id. */
  hypothesisId: string;
  text: string;
  confidence: number;
}

export interface DecisionEvent extends TraceEventBase {
  type: "DECISION";
  level: "DERIVED";
  decision: "sampled" | "greedy" | "stop" | "max_tokens" | "aborted";
  detail?: string;
}

export interface UncertaintyEvent extends TraceEventBase {
  type: "UNCERTAINTY";
  level: "DERIVED";
  kind:
    | "MODEL_UNCERTAINTY"
    | "EVIDENCE_QUALITY"
    | "INPUT_AMBIGUITY"
    | "ANSWER_STABILITY";
  value: number;
  window?: { fromStep: number; toStep: number };
}

export interface OutputEvent extends TraceEventBase {
  type: "OUTPUT";
  level: "MEASURED";
  text: string;
  tokenCount: number;
  durationMs: number;
  finishReason: string;
}

export type TraceEvent =
  | InputEvent
  | TokenEvent
  | LayerActivityEvent
  | AttentionEvent
  | ConceptEvent
  | EvidenceEvent
  | HypothesisEvent
  | DecisionEvent
  | UncertaintyEvent
  | OutputEvent;

export interface TraceModelInfo {
  name: string;
  revision: string;
  device: string;
  layerCount: number;
  paramCount: number;
}

export interface TraceSampling {
  maxTokens: number;
  temperature: number;
  topK: number | null;
  seed: number | null;
}

export interface TraceOutput {
  text: string;
  tokenCount: number;
  durationMs: number;
  finishReason: string;
}

/**
 * The trace envelope. During a live stream `events` is empty — events arrive
 * over SSE; on GET /trace/{id} and in fixtures, `events` is the full ordered
 * history (replay source of truth).
 */
export interface Trace {
  /** `"tr_" + 8-char base36`. */
  id: string;
  /** DB serial, rendered as "TRACE 0042". */
  displayId: number;
  model: TraceModelInfo;
  input: { text: string };
  traceMode: TraceMode;
  sampling: TraceSampling;
  status: TraceStatus;
  createdAt: string;
  output?: TraceOutput;
  events: TraceEvent[];
}
