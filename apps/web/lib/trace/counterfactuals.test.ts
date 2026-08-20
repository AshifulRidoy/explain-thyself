/**
 * The counterfactual panel's pure core: the VARIABLE/IMPACT ranking
 * (spec §23) and the shared contract it parses against.
 */
import { describe, expect, it } from "vitest";
import {
  applicableSubstitutions,
  counterfactualResultSchema,
  CUSTOM_VARIABLE,
  type CounterfactualResult,
} from "@ets/trace-schema";
import { rankByImpact } from "./counterfactuals";

function result(over: Partial<CounterfactualResult>): CounterfactualResult {
  return counterfactualResultSchema.parse({
    id: "cf_abc12345",
    traceId: "tr_abc12345",
    variable: "experience",
    originalWord: "beginner",
    replacementWord: "veteran",
    promptText: "I am a veteran.",
    outputText: "an answer",
    tokenCount: 10,
    agreedTokens: 4,
    impact: 0.6,
    firstDivergence: 2,
    entropyDelta: -0.3,
    basis: "greedy rerun …",
    createdAt: "2026-08-19T00:00:00.000Z",
    ...over,
  });
}

describe("rankByImpact (spec §23)", () => {
  it("sorts by measured impact descending — the biggest mover tops the table", () => {
    const ranked = rankByImpact([
      result({ id: "cf_low00001", impact: 0.1, promptText: "a" }),
      result({ id: "cf_high0001", impact: 0.9, promptText: "b" }),
      result({ id: "cf_mid00001", impact: 0.5, promptText: "c" }),
    ]);
    expect(ranked.map((r) => r.id)).toEqual(["cf_high0001", "cf_mid00001", "cf_low00001"]);
  });

  it("a rerun of the same edit supersedes its earlier measurement", () => {
    const ranked = rankByImpact([
      result({ id: "cf_first001", promptText: "same edit", impact: 0.9 }),
      result({ id: "cf_second01", promptText: "same edit", impact: 0.2 }),
    ]);
    expect(ranked).toHaveLength(1);
    expect(ranked[0].id).toBe("cf_second01");
  });

  it("dictionary variables and free-form edits rank together", () => {
    const ranked = rankByImpact([
      result({ variable: "experience", impact: 0.3, promptText: "a" }),
      result({
        variable: CUSTOM_VARIABLE,
        originalWord: null,
        replacementWord: null,
        impact: 0.8,
        promptText: "b",
      }),
    ]);
    expect(ranked[0].variable).toBe(CUSTOM_VARIABLE);
  });
});

describe("applicableSubstitutions (shared contract)", () => {
  it("resolves whole words, preserves case, one variable per run", () => {
    expect(
      applicableSubstitutions("Why is the sky blue?").map((v) => v.promptText),
    ).toEqual(["How is the sky blue?", "Why is the ocean blue?", "Why is the sky green?"]);
  });

  it("prompts with no dictionary words propose nothing — honestly", () => {
    expect(applicableSubstitutions("Tell me about tensors.")).toEqual([]);
  });
});
