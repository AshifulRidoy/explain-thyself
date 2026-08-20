/**
 * Cross-model comparison (spec Phase 7, pulled into V2 by the Roadmap):
 * "run the same prompt through multiple registered models and compare
 * the traces."
 *
 * One comparison = one prompt, two REGISTERED models, two recorded
 * traces. Model B's run is a full persisted trace (its own replay, its
 * own events); the comparison itself is a separate artifact attached to
 * trace A — the same immutability rule as counterfactuals. Never an
 * event appended to either trace.
 *
 * The one hard precondition: the two models must share a tokenizer
 * (`ModelSpec.tokenizer`), otherwise token IDS are not comparable and
 * "agreement" would be a number about nothing. The engine rejects the
 * request (400) rather than compute it; fake word-hash ids only ever
 * compare against the fake.
 *
 *   agreement = agreed/comparedLength   DERIVED — shared-token overlap
 *                                       over the first min(lenA, lenB)
 *                                       positions; each answer keeps
 *                                       its own length (tokenCountA/B)
 *   ΔH        = mean entropy shift      DERIVED, signed — from the
 *                                       SHIPPED entropyBits of both
 *                                       traces, so it is recomputable
 *                                       from the trace JSON
 *
 * What agreement is NOT: internal similarity. It is surface token
 * overlap under a common tokenizer; the models' internals are compared
 * only through their own traces.
 *
 * Mirrored 1:1 by services/trace-engine/app/schemas/trace.py — keep
 * both sides in sync.
 */

import { z } from "zod";

/** One completed comparison (API artifact, not a TraceEvent). */
export const comparisonResultSchema = z
  .object({
    id: z.string().regex(/^cmp_[0-9a-z]{6,12}$/),
    /** the anchor trace — the one this comparison hangs off */
    traceIdA: z.string().regex(/^tr_/),
    /** the freshly recorded run through model B */
    traceIdB: z.string().regex(/^tr_/),
    modelA: z.string().min(1),
    modelB: z.string().min(1),
    /** the prompt both models answered */
    prompt: z.string().min(1),
    /** each answer's own emitted token count */
    tokenCountA: z.number().int().positive(),
    tokenCountB: z.number().int().positive(),
    /** min(tokenCountA, tokenCountB) — the positions compared */
    comparedLength: z.number().int().positive(),
    agreedTokens: z.number().int().min(0),
    /** agreed/comparedLength; 1 = identical over the compared range */
    agreement: z.number().min(0).max(1),
    /** first compared position where the token ids differ; null when identical */
    firstDivergence: z.number().int().min(0).nullable(),
    outputTextA: z.string(),
    outputTextB: z.string(),
    /** mean shipped entropyBits of each answer (bits/token) */
    meanEntropyA: z.number().min(0),
    meanEntropyB: z.number().min(0),
    /** meanB − meanA, signed */
    entropyDelta: z.number(),
    basis: z.string().min(1),
    createdAt: z.string().min(1),
  })
  .strict();

export type ComparisonResult = z.infer<typeof comparisonResultSchema>;

export const comparisonRequestSchema = z
  .object({
    /** registry key of the model to run the prompt through (model B) */
    model: z.string().min(1),
  })
  .strict();

export type ComparisonRequest = z.infer<typeof comparisonRequestSchema>;
