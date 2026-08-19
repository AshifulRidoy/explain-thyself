/**
 * The concept dictionary — Phase 5's INTERPRETED layer (spec §20, §38).
 *
 * A concept is a HUMAN-AUTHORED set of vocabulary words. Its score on a
 * step is exact: the probability mass the model's full next-token
 * distribution places on those words' token ids. The mass is MEASURED;
 * the claim that the word set deserves its label is the INTERPRETATION.
 * Nothing here pretends the model contains a neuron named after a label.
 *
 * This file is the single authored source. The engine mirrors it in
 * services/trace-engine/app/aggregation/concepts.py — keep them in sync
 * (the fixture corpus exercises both, and tests assert no word appears
 * in two concepts).
 *
 * Dictionary rules:
 *   - every word must be a single GPT-2 token in the leading-space form
 *     (" word") — the form that realizes the word in running text, which
 *     is where next-token mass lives (model-marked tests reject anything
 *     else — an unmatchable word silently scores 0 forever). The bare
 *     form also resolves when it is single-token (~2/3 of words; GPT-2
 *     merges are space-prefixed), and the engine scores both ids.
 *   - disjoint word sets: mass must be attributable to exactly one label
 *   - labels are lowercase noun phrases; conceptIds are snake_case
 */

export interface ConceptSpec {
  conceptId: string;
  label: string;
  words: string[];
}

/** Emit a CONCEPT event when the concept's mass reaches this level. */
export const CONCEPT_ACTIVE_MASS = 0.05;

/** How many matched tokens to ship as auditable evidence per event. */
export const CONCEPT_EVIDENCE_LIMIT = 3;

export const CONCEPT_DICTIONARY: ConceptSpec[] = [
  {
    conceptId: "concept_uncertainty",
    label: "uncertainty / hedging",
    words: [
      "maybe", "perhaps", "might", "could", "possibly", "probably",
      "likely", "unclear", "uncertain", "depends", "sometimes",
      "generally", "usually", "often", "seems",
    ],
  },
  {
    conceptId: "concept_certainty",
    label: "certainty / emphasis",
    words: [
      "definitely", "certainly", "always", "must", "clearly", "obviously",
      "truly", "certain", "surely", "indeed", "simply", "exactly",
    ],
  },
  {
    conceptId: "concept_comparison",
    label: "comparison / choice",
    words: [
      "better", "worse", "best", "worst", "versus", "compared",
      "difference", "similar", "instead", "alternative", "prefer",
      "choice", "trade", "options",
    ],
  },
  {
    conceptId: "concept_causality",
    label: "causality / explanation",
    words: [
      "because", "since", "therefore", "thus", "causes", "caused",
      "leads", "result", "results", "hence", "due", "reason", "why",
      "explains", "effect",
    ],
  },
  {
    conceptId: "concept_agency",
    label: "people / agency",
    words: [
      "people", "person", "user", "human", "humans", "everyone",
      "someone", "readers", "students", "developers", "workers",
    ],
  },
  {
    conceptId: "concept_quantity",
    label: "quantity / degree",
    words: [
      "many", "few", "most", "several", "number", "numbers", "percent",
      "half", "twice", "majority", "minority", "lots", "plenty",
    ],
  },
  {
    conceptId: "concept_time",
    label: "time / change",
    words: [
      "today", "now", "soon", "later", "eventually", "years", "year",
      "decade", "century", "past", "future", "history", "before",
      "after", "changes", "changed",
    ],
  },
  {
    conceptId: "concept_technology",
    label: "technology / systems",
    words: [
      "computer", "computers", "software", "code", "data", "system",
      "systems", "model", "models", "machine", "machines", "digital",
      "program", "programs", "network", "tool", "tools", "hardware",
    ],
  },
  {
    conceptId: "concept_science",
    label: "science / nature",
    words: [
      "light", "energy", "climate", "weather", "air", "water", "earth",
      "physics", "science", "biology", "natural", "molecules", "atoms",
      "chemical", "sun", "sunlight", "color", "colors", "waves",
    ],
  },
  {
    conceptId: "concept_language",
    label: "language / learning",
    words: [
      "language", "languages", "word", "words", "meaning", "learn",
      "learning", "read", "reading", "write", "writing", "text",
      "sentence", "grammar", "syntax", "practice", "teaching",
    ],
  },
];
