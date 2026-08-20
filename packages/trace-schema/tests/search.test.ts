/**
 * Trace-search contract (spec §28): the response schema both runtimes
 * share, and the honesty rules baked into it (basis disclaims meaning,
 * similarity bounded, empty results legal).
 */
import { describe, expect, it } from "vitest";
import {
  SEARCH_BASIS,
  searchHitSchema,
  searchResponseSchema,
} from "../src/search.js";

const hit = {
  traceId: "tr_abc12345",
  displayId: 42,
  input: "Why is the sky blue?",
  similarity: 0.9812,
  modelName: "gpt2-small",
  traceMode: "STANDARD",
  tokenCount: 30,
  createdAt: "2026-08-20T00:00:00.000Z",
};

describe("searchHitSchema", () => {
  it("accepts a well-formed hit and rejects unknown fields", () => {
    expect(searchHitSchema.parse(hit)).toEqual(hit);
    expect(searchHitSchema.safeParse({ ...hit, extra: 1 }).success).toBe(false);
  });

  it("bounds similarity to [-1, 1] and requires a tr_ trace", () => {
    expect(searchHitSchema.safeParse({ ...hit, similarity: 1.4 }).success).toBe(false);
    expect(searchHitSchema.safeParse({ ...hit, similarity: -1 }).success).toBe(true);
    expect(searchHitSchema.safeParse({ ...hit, traceId: "xx_1" }).success).toBe(false);
  });
});

describe("searchResponseSchema", () => {
  it("accepts an empty result set — no matches is a valid answer", () => {
    const parsed = searchResponseSchema.parse({
      query: "quantum wombats",
      basis: SEARCH_BASIS,
      results: [],
      searchable: 431,
    });
    expect(parsed.results).toEqual([]);
  });

  it("rejects a missing basis — the UI must never caption the number itself", () => {
    const { basis: _basis, ...withoutBasis } = {
      query: "q",
      basis: SEARCH_BASIS,
      results: [],
      searchable: 0,
    };
    expect(searchResponseSchema.safeParse(withoutBasis).success).toBe(false);
  });
});

describe("SEARCH_BASIS", () => {
  it("keeps disclaiming semantic meaning and naming the anisotropy", () => {
    expect(SEARCH_BASIS).toContain("not semantic meaning");
    expect(SEARCH_BASIS).toContain("anisotropic");
    expect(SEARCH_BASIS).toContain("rank");
  });
});
