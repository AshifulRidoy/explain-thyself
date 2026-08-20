/**
 * Counterfactual mode (spec §23 / Phase 6): "WHAT WOULD CHANGE THE
 * ANSWER?" — controlled prompt variations, rerun, compared.
 *
 * A *variable* is one word-level substitution from this human-authored
 * dictionary (spec §26 would use an LLM app layer to generate
 * counterfactual questions; this instrument has no such layer, so the
 * variables come from a curated in-repo dictionary — the same pattern as
 * the concept dictionary: the label is INTERPRETED, the numbers under it
 * are measured).
 *
 * Each applicable substitution reruns the model greedy on the edited
 * prompt and compares the answer token-by-token against the original:
 *
 *   impact   = 1 − agreed/total      DERIVED — 0 means the answer is
 *                                    byte-identical under the edit
 *   ΔH       = mean entropy shift    DERIVED, signed — did the edit make
 *                                    the model more or less uncertain
 *
 * What impact is NOT: causal attribution. It is word-substitution
 * sensitivity at the prompt level; activation patching (the causal
 * version) is Phase 7 / V3.
 *
 * Mirrored 1:1 by services/trace-engine/app/aggregation/counterfactuals.py
 * — keep both sides in sync (same dictionary, same applicability rules).
 */

import { z } from "zod";

/** One dictionary entry: swapping `word` for `replacement` manipulates `variable`. */
export interface CounterfactualEntry {
  /** lowercase lookup word, matched case-insensitively at word boundaries */
  word: string;
  replacement: string;
  /** INTERPRETED label shown in the VARIABLE column (spec §23) */
  variable: string;
}

export const COUNTERFACTUAL_DICTIONARY: readonly CounterfactualEntry[] = [
  { word: "beginner", replacement: "veteran", variable: "experience" },
  { word: "expert", replacement: "beginner", variable: "experience" },
  { word: "simple", replacement: "complex", variable: "task complexity" },
  { word: "easy", replacement: "hard", variable: "difficulty" },
  { word: "fast", replacement: "slow", variable: "performance" },
  { word: "slow", replacement: "fast", variable: "performance" },
  { word: "quickly", replacement: "carefully", variable: "learning speed" },
  { word: "python", replacement: "rust", variable: "language" },
  { word: "rust", replacement: "python", variable: "language" },
  { word: "hobby", replacement: "job", variable: "career goal" },
  { word: "why", replacement: "how", variable: "question type" },
  { word: "sky", replacement: "ocean", variable: "subject" },
  { word: "blue", replacement: "green", variable: "subject" },
];

/** Cap on substitutions per "investigate" run — each one is a full rerun. */
export const MAX_COUNTERFACTUALS = 6;

/** An applicable dictionary entry, resolved against a concrete prompt. */
export interface CounterfactualVariable {
  /** INTERPRETED label, e.g. "experience" */
  variable: string;
  /** the word as it actually appears in the prompt, e.g. "Python" */
  originalWord: string;
  /** the replacement, case-matched to the original, e.g. "Rust" */
  replacementWord: string;
  /** the edited prompt this variable would rerun */
  promptText: string;
}

function escapeRegex(word: string): string {
  return word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Match the original's capitalization: "Python" → "Rust", "python" → "rust". */
function matchCase(replacement: string, original: string): string {
  const first = original[0];
  if (first === first.toUpperCase() && first !== first.toLowerCase()) {
    return replacement[0].toUpperCase() + replacement.slice(1);
  }
  return replacement;
}

/**
 * Every dictionary word that appears in the prompt (whole-word,
 * case-insensitive, first occurrence), in dictionary order, capped at
 * MAX_COUNTERFACTUALS. Each becomes ONE single-variable rerun — swapping
 * one word at a time is what makes the impact attributable to a variable.
 */
export function applicableSubstitutions(
  prompt: string,
): CounterfactualVariable[] {
  const found: CounterfactualVariable[] = [];
  for (const entry of COUNTERFACTUAL_DICTIONARY) {
    if (found.length >= MAX_COUNTERFACTUALS) break;
    const match = new RegExp(`\\b${escapeRegex(entry.word)}\\b`, "i").exec(
      prompt,
    );
    if (!match) continue;
    const replacement = matchCase(entry.replacement, match[0]);
    found.push({
      variable: entry.variable,
      originalWord: match[0],
      replacementWord: replacement,
      promptText:
        prompt.slice(0, match.index) +
        replacement +
        prompt.slice(match.index + match[0].length),
    });
  }
  return found;
}

/** Label for free-form (user-edited) counterfactuals — not a dictionary variable. */
export const CUSTOM_VARIABLE = "your edit";

/** One completed counterfactual comparison (API artifact, not a TraceEvent —
 * the original trace stays immutable; results persist in `counterfactuals`). */
export const counterfactualResultSchema = z
  .object({
    id: z.string().regex(/^cf_[0-9a-z]{6,12}$/),
    traceId: z.string().regex(/^tr_/),
    /** INTERPRETED variable label, or "your edit" for a free-form rerun */
    variable: z.string().min(1),
    /** null for free-form edits (no single word was manipulated) */
    originalWord: z.string().min(1).nullable(),
    replacementWord: z.string().min(1).nullable(),
    /** the edited prompt that was rerun */
    promptText: z.string().min(1),
    /** the answer the edited prompt produced */
    outputText: z.string(),
    /** compared length = the ORIGINAL trace's emitted token count */
    tokenCount: z.number().int().positive(),
    agreedTokens: z.number().int().min(0),
    /** 1 − agreed/tokenCount; 0 = byte-identical answer under the edit */
    impact: z.number().min(0).max(1),
    /** first step where the answers differ; null when identical */
    firstDivergence: z.number().int().min(0).nullable(),
    /** signed mean entropy shift vs. the original answer (bits/token) */
    entropyDelta: z.number(),
    basis: z.string().min(1),
    createdAt: z.string().min(1),
  })
  .strict();

export type CounterfactualResult = z.infer<typeof counterfactualResultSchema>;

export const counterfactualRequestSchema = z
  .object({
    /** all — run every applicable dictionary substitution (capped);
     *  one — rerun a specific resolved variable; prompt — free-form edit */
    scope: z.enum(["all", "one", "prompt"]),
    variable: z.string().optional(),
    originalWord: z.string().optional(),
    prompt: z.string().min(1).max(2000).optional(),
  })
  .strict();

export type CounterfactualRequest = z.infer<typeof counterfactualRequestSchema>;
