/**
 * The uncertainty layer (spec §22): four SEPARATE quantities, never one
 * blended "AI confidence" number.
 *
 *   MODEL_UNCERTAINTY  DERIVED    mean normalized entropy H/log2(V) over
 *                                  the emitted tokens — from the full
 *                                  next-token distribution the trace
 *                                  already measured
 *   ANSWER_STABILITY   MEASURED   greedy token agreement across
 *                                  controlled surface perturbations of
 *                                  the prompt (rerun after the answer)
 *   INPUT_AMBIGUITY    —          null: needs an auxiliary model this
 *                                  instrument does not have
 *   EVIDENCE_QUALITY   —          null: no retrieval sources attached
 *
 * The two null quantities ship as events with `value: null` and the
 * reason in `basis` — the refusal is part of the trace, not UI copy.
 *
 * Mirrored 1:1 by services/trace-engine/app/aggregation/stability.py —
 * keep both sides in sync (same perturbations, same constants).
 */

/** Deterministic surface perturbations of a prompt (semantics-preserving). */
export interface PromptPerturbation {
  /** machine name, e.g. "lowercase_first" */
  name: string;
  /** the perturbed prompt actually rerun */
  text: string;
}

/**
 * The four quantities in spec §22's display order — emission and UI both
 * render in this order. Mirrored by UNCERTAINTY_KINDS in stability.py.
 */
export const UNCERTAINTY_KINDS = [
  "MODEL_UNCERTAINTY",
  "EVIDENCE_QUALITY",
  "INPUT_AMBIGUITY",
  "ANSWER_STABILITY",
] as const;

export function promptPerturbations(text: string): PromptPerturbation[] {
  const out: PromptPerturbation[] = [];
  const trimmed = text.trimEnd();
  if (trimmed.length > 1 && /[.?!]$/.test(trimmed)) {
    out.push({ name: "strip_final_punct", text: trimmed.slice(0, -1) });
  }
  if (/^[A-Z]/.test(text)) {
    out.push({ name: "lowercase_first", text: text[0].toLowerCase() + text.slice(1) });
  }
  return out;
}

/** Fake-vocab size shared by the fixture generator and FakeBackend. */
export const FAKE_VOCAB_SIZE = 50_000;

/** Seed for one fake perturbed run — FNV-1a over the perturbed prompt. */
export function textSeed(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Replacement words for fake perturbed runs — plausible running text, so a
 * diverging fake trace still reads like language. Mirrored in stability.py.
 */
export const FAKE_SWAP_POOL: readonly string[] = [
  "notably", "broadly", "quickly", "largely", "partly", "slowly",
  "widely", "deeply", "often", "usually", "gently", "readily",
];

/** Chance a content word is swapped in a fake perturbed run. */
export const FAKE_SWAP_RATE = 0.18;
