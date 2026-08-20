/**
 * Cross-model comparison contract (spec Phase 7, V2 cut): the artifact
 * schema both runtimes share, and the honesty rules baked into it —
 * agreement is bounded, both token counts ship next to it (length
 * differences are visible, not folded into the score), and the basis
 * field is mandatory so the UI never captions the number itself.
 */
import { describe, expect, it } from "vitest";
import {
  comparisonRequestSchema,
  comparisonResultSchema,
} from "../src/comparison.js";

const result = {
  id: "cmp_ab12cd34",
  traceIdA: "tr_anchor01",
  traceIdB: "tr_other002",
  modelA: "gpt2-small",
  modelB: "distilgpt2",
  prompt: "Why is the sky blue?",
  tokenCountA: 12,
  tokenCountB: 12,
  comparedLength: 12,
  agreedTokens: 2,
  agreement: 0.1667,
  firstDivergence: 2,
  outputTextA: "  The sky blue is a color",
  outputTextB: "            ",
  meanEntropyA: 5.8662,
  meanEntropyB: 5.6002,
  entropyDelta: -0.266,
  basis:
    "same prompt rerun greedy through distilgpt2 (12 tokens) and compared " +
    "position-by-position against gpt2-small's recorded answer (12 tokens) " +
    "over the first 12 positions; agreement is shared token overlap under a " +
    "common tokenizer — surface behavior, not internal similarity",
  createdAt: "2026-08-20T00:00:00.000Z",
};

describe("comparisonResultSchema", () => {
  it("accepts a well-formed comparison and rejects unknown fields", () => {
    expect(comparisonResultSchema.parse(result)).toEqual(result);
    expect(
      comparisonResultSchema.safeParse({ ...result, extra: 1 }).success,
    ).toBe(false);
  });

  it("bounds agreement to [0, 1] and requires cmp_/tr_ ids", () => {
    expect(
      comparisonResultSchema.safeParse({ ...result, agreement: 1.2 }).success,
    ).toBe(false);
    expect(
      comparisonResultSchema.safeParse({ ...result, id: "cf_ab12cd34" }).success,
    ).toBe(false);
    expect(
      comparisonResultSchema.safeParse({ ...result, traceIdB: "xx_1" }).success,
    ).toBe(false);
  });

  it("allows firstDivergence null (identical over the compared range)", () => {
    const parsed = comparisonResultSchema.parse({
      ...result,
      agreedTokens: 12,
      agreement: 1,
      firstDivergence: null,
    });
    expect(parsed.firstDivergence).toBeNull();
  });

  it("requires positive token counts — an answer with no tokens is not comparable", () => {
    expect(
      comparisonResultSchema.safeParse({ ...result, tokenCountB: 0 }).success,
    ).toBe(false);
    expect(
      comparisonResultSchema.safeParse({ ...result, comparedLength: 0 }).success,
    ).toBe(false);
  });

  it("keeps a signed entropy delta — direction is the finding", () => {
    const parsed = comparisonResultSchema.parse({
      ...result,
      entropyDelta: -1.25,
    });
    expect(parsed.entropyDelta).toBeLessThan(0);
  });
});

describe("comparisonRequestSchema", () => {
  it("takes exactly one model key", () => {
    expect(comparisonRequestSchema.parse({ model: "distilgpt2" })).toEqual({
      model: "distilgpt2",
    });
    expect(comparisonRequestSchema.safeParse({ model: "" }).success).toBe(false);
    expect(comparisonRequestSchema.safeParse({}).success).toBe(false);
    expect(
      comparisonRequestSchema.safeParse({ model: "x", scope: "all" }).success,
    ).toBe(false);
  });
});
