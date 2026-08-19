/**
 * The uncertainty layer's UI helpers (spec §22): display names for the
 * four separated quantities and the selector that keeps them in the
 * spec's display order (the engine emits them in that order).
 */
import type { TraceEvent, UncertaintyEvent } from "@ets/trace-schema";

export const UNCERTAINTY_LABELS: Record<UncertaintyEvent["kind"], string> = {
  MODEL_UNCERTAINTY: "model uncertainty",
  EVIDENCE_QUALITY: "evidence quality",
  INPUT_AMBIGUITY: "input ambiguity",
  ANSWER_STABILITY: "answer stability",
};

/** All UNCERTAINTY events in emission order — the panel renders in rows. */
export function uncertaintyQuantities(events: TraceEvent[]): UncertaintyEvent[] {
  return events.filter((e): e is UncertaintyEvent => e.type === "UNCERTAINTY");
}

/** MODEL_UNCERTAINTY → "modelUncertainty" — the SIGNAL_TAXONOMY key suffix. */
export function uncertaintyTaxonomyKey(kind: UncertaintyEvent["kind"]): string {
  return kind.toLowerCase().replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
}
