/**
 * Trace search (spec §28 / V2): "find traces similar to this one" —
 * pgvector cosine similarity over recorded traces.
 *
 * The embedding is the MODEL'S OWN representation of the prompt, not an
 * external embedder's: the mean of the final-layer residual stream
 * (resid_post) over the prompt's tokens, L2-normalized — 768 floats for
 * GPT-2 small. The engine computes it in one forward pass when a trace
 * starts and stores it in `traces.embedding`; search is cosine distance
 * in Postgres (`<=>`), exact scan (no ANN index at this scale).
 *
 * Honesty rules this module exists to enforce:
 *  - similarity is DERIVED (a cosine between two measured vectors);
 *    "these prompts mean the same thing" is an INTERPRETATION nobody
 *    makes — the ranking says "the model represents these prompts
 *    closely in its final layer", nothing more
 *  - GPT-2's hidden space is anisotropic: absolute cosines sit high and
 *    compressed; the RANK is the signal, the number is not a confidence
 *  - rows recorded before this column (or whose embedding pass failed)
 *    are NULL and simply unsearchable — never silently approximated
 *
 * Mirrored by services/trace-engine/app/schemas/trace.py (response
 * models) — keep both sides in sync.
 */

import { z } from "zod";

/** Dimensionality of the stored vector (GPT-2 small d_model; fake matches). */
export const EMBEDDING_DIM = 768;

/** One ranked match. `basis` states what the similarity is and is not. */
export interface SearchHit {
  traceId: string;
  /** DB serial rendered as "TRACE 0042" */
  displayId: number;
  input: string;
  /** cosine similarity ∈ [-1, 1]; with a normalized vector this is
   *  1 − cosine_distance from pgvector, rounded to 4 decimals */
  similarity: number;
  modelName: string;
  traceMode: string;
  tokenCount: number | null;
  createdAt: string;
}

export const searchHitSchema = z
  .object({
    traceId: z.string().regex(/^tr_/),
    displayId: z.number().int().positive(),
    input: z.string().min(1),
    similarity: z.number().min(-1).max(1),
    modelName: z.string().min(1),
    traceMode: z.enum(["BASIC", "STANDARD", "RESEARCH"]),
    tokenCount: z.number().int().min(0).nullable(),
    createdAt: z.string().min(1),
  })
  .strict();

export type SearchHitParsed = z.infer<typeof searchHitSchema>;

/** Response of GET /search?q= and GET /trace/{id}/similar. */
export const searchResponseSchema = z
  .object({
    /** the free-text query, or the source trace's prompt for /similar */
    query: z.string(),
    /** fixed, so the UI never invents its own caption for the number */
    basis: z.string().min(1),
    /** matches already ranked (similarity desc); empty is a valid answer */
    results: z.array(searchHitSchema),
    /** how many stored traces carry an embedding and were compared */
    searchable: z.number().int().min(0),
  })
  .strict();

export type SearchResponse = z.infer<typeof searchResponseSchema>;

/** What a search result's similarity means — one string, both runtimes. */
export const SEARCH_BASIS =
  "cosine similarity of the model's final-layer prompt representation " +
  "(mean resid_post over prompt tokens) — rank orders how the model " +
  "represents these prompts; it is not semantic meaning, and GPT-2's " +
  "hidden space is anisotropic, so absolute values are compressed";

/** GET /search query parameters (all optional; limit is clamped 1..50). */
export const searchRequestSchema = z
  .object({
    q: z.string().min(1).max(2000),
    limit: z.number().int().min(1).max(50).optional(),
  })
  .strict();

export type SearchRequest = z.infer<typeof searchRequestSchema>;
