import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  FIXTURES,
  generateTraceFixture,
  mulberry32,
} from "../src/generate.js";
import { CONCEPT_ACTIVE_MASS, CONCEPT_DICTIONARY } from "../src/concepts.js";
import { traceSchema, validateTrace } from "../src/schema.js";
import { SIGNAL_TAXONOMY } from "../src/taxonomy.js";

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

async function readFixture(key: string): Promise<unknown> {
  return JSON.parse(await readFile(join(FIXTURES_DIR, `${key}.json`), "utf8"));
}

/** Minimal valid envelope for parse-level event tests. */
function baseTrace() {
  return {
    id: "tr_abc12345",
    displayId: 1,
    model: {
      name: "gpt2-small", revision: "", device: "cpu", layerCount: 12,
      paramCount: 124000000,
    },
    input: { text: "x" },
    traceMode: "STANDARD",
    sampling: { maxTokens: 4, temperature: 0, topK: null, seed: 1 },
    status: "complete",
    createdAt: "2026-08-16T00:00:00.000Z",
    events: [],
  };
}

describe("generator determinism", () => {
  it("same seed ⇒ byte-identical JSON", () => {
    const spec = FIXTURES[0];
    const a = JSON.stringify(generateTraceFixture(spec));
    const b = JSON.stringify(generateTraceFixture(spec));
    expect(a).toBe(b);
  });

  it("mulberry32 is stable for a known seed", () => {
    // pinned: the Python FakeBackend port must produce this same value
    const rng = mulberry32(42);
    expect(rng()).toBeCloseTo(0.6011037519201636, 12);
  });
});

describe("committed fixtures validate against the zod schema", () => {
  for (const spec of FIXTURES) {
    it(`${spec.key}.json is a valid Trace`, async () => {
      const trace = validateTrace(await readFixture(spec.key));
      expect(trace.id).toMatch(/^tr_/);
    });
  }

  it("regenerating matches the committed files byte-for-byte", async () => {
    for (const spec of FIXTURES) {
      const onDisk = await readFile(join(FIXTURES_DIR, `${spec.key}.json`), "utf8");
      const regenerated = JSON.stringify(generateTraceFixture(spec), null, 2) + "\n";
      expect(regenerated).toBe(onDisk);
    }
  });
});

describe("fixture realism invariants", () => {
  it("seq is gapless and t is monotone", () => {
    const trace = generateTraceFixture(FIXTURES[1]);
    trace.events.forEach((e, i) => expect(e.seq).toBe(i));
    for (let i = 1; i < trace.events.length; i++) {
      expect(trace.events[i].t).toBeGreaterThanOrEqual(trace.events[i - 1].t);
    }
  });

  it("entropy/probability stay coupled (p tracks 2^-H within jitter)", () => {
    const trace = generateTraceFixture(FIXTURES[0]);
    const tokens = trace.events.filter((e) => e.type === "TOKEN");
    expect(tokens.length).toBeGreaterThan(30);
    for (const tok of tokens) {
      if (tok.type !== "TOKEN") continue;
      const implied = 2 ** -tok.entropyBits;
      expect(tok.probability).toBeGreaterThan(0);
      // greedy argmax probability must be at least the uniform-implies level
      // and topK is non-increasing (tiny tails round to equal zeros)
      expect(tok.probability).toBeGreaterThan(implied * 0.5);
      for (let i = 1; i < tok.topK.length; i++) {
        expect(tok.topK[i].probability).toBeLessThanOrEqual(tok.topK[i - 1].probability);
      }
      // alternatives must never beat the argmax
      for (const cand of tok.topK.slice(1)) {
        expect(cand.probability).toBeLessThan(tok.topK[0].probability);
      }
    }
  });

  it("function words get lower entropy than content words on average", () => {
    const trace = generateTraceFixture(FIXTURES[1]);
    const tokens = trace.events.filter(
      (e): e is Extract<typeof e, { type: "TOKEN" }> => e.type === "TOKEN",
    );
    const fn = tokens.filter((t) => t.entropyBits < 1).map((t) => t.entropyBits);
    const content = tokens.filter((t) => t.entropyBits >= 1.2).map((t) => t.entropyBits);
    const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
    expect(avg(content)).toBeGreaterThan(avg(fn));
  });

  it("l2Norm grows with depth and normRatio starts at 1.0", () => {
    const trace = generateTraceFixture(FIXTURES[1]);
    const layers = trace.events.filter(
      (e): e is Extract<typeof e, { type: "LAYER_ACTIVITY" }> => e.type === "LAYER_ACTIVITY",
    );
    const first = layers[0].layers;
    for (const l of first) expect(l.normRatio).toBe(1);
    const last = layers[layers.length - 1].layers;
    expect(last[last.length - 1].l2Norm).toBeGreaterThan(last[0].l2Norm);
  });
});

describe("attention fixtures (Phase 4)", () => {
  const skyBlue = FIXTURES.find((f) => f.key === "trace-sky-blue")!;

  it("only the attention fixture is RESEARCH; others emit no ATTENTION events", () => {
    const research = generateTraceFixture(skyBlue);
    expect(research.traceMode).toBe("RESEARCH");
    expect(research.events.some((e) => e.type === "ATTENTION")).toBe(true);

    for (const spec of FIXTURES.filter((f) => !f.attention)) {
      const trace = generateTraceFixture(spec);
      expect(trace.traceMode).toBe("STANDARD");
      expect(trace.events.some((e) => e.type === "ATTENTION")).toBe(false);
    }
  });

  it("emits one ATTENTION per layer per step, layers 1..N in order", () => {
    const trace = generateTraceFixture(skyBlue);
    const tokens = trace.events.filter((e) => e.type === "TOKEN");
    const attn = trace.events.filter(
      (e): e is Extract<typeof e, { type: "ATTENTION" }> => e.type === "ATTENTION",
    );
    expect(attn.length).toBe(tokens.length * skyBlue.layerCount);
    // consecutive blocks of layerCount, ordered 1..N
    for (let s = 0; s < tokens.length; s++) {
      const block = attn.slice(s * skyBlue.layerCount, (s + 1) * skyBlue.layerCount);
      expect(block.map((e) => e.layer)).toEqual(
        Array.from({ length: skyBlue.layerCount }, (_, i) => i + 1),
      );
      for (const e of block) expect(e.position).toBe(tokens[s].position);
    }
  });

  it("rows are distributions over [BOS, prompt, generated-so-far]", () => {
    const trace = generateTraceFixture(skyBlue);
    const promptLen = trace.events[0].type === "INPUT" ? trace.events[0].tokenCount : 0;
    const attn = trace.events.filter(
      (e): e is Extract<typeof e, { type: "ATTENTION" }> => e.type === "ATTENTION",
    );
    for (let i = 0; i < attn.length; i++) {
      const e = attn[i];
      const step = Math.floor(i / skyBlue.layerCount);
      const row = e.aggregated!;
      // BOS + prompt + tokens emitted before this step
      expect(row.length).toBe(promptLen + step + 1);
      // BOS is entry 0, surfaced as position -1
      expect(row[0]).toMatchObject({ position: -1, text: "<bos>" });
      expect(row[1].position).toBe(0);
      // head-mean of softmax rows: a distribution (4-dp rounding slack)
      const sum = row.reduce((a, r) => a + r.weight, 0);
      expect(Math.abs(sum - 1)).toBeLessThan(5e-3);
      // per-head entropies: 12 heads, each bounded by log2(row length)
      expect(e.headEntropyBits!.length).toBe(12);
      for (const h of e.headEntropyBits!) {
        expect(h).toBeGreaterThan(0);
        expect(h).toBeLessThanOrEqual(Math.log2(row.length) + 1e-3);
      }
    }
  });

  it("BOS sink grows with depth (mirrors real GPT-2)", () => {
    const trace = generateTraceFixture(skyBlue);
    const attn = trace.events.filter(
      (e): e is Extract<typeof e, { type: "ATTENTION" }> => e.type === "ATTENTION",
    );
    const bos = (layerIdx: number) => attn[layerIdx].aggregated![0].weight;
    expect(bos(skyBlue.layerCount - 1)).toBeGreaterThan(bos(0));
    expect(bos(skyBlue.layerCount - 1)).toBeGreaterThan(0.1);
  });
});

describe("concept fixtures (Phase 5)", () => {
  it("every fixture's CONCEPTs are dictionary-scored, positioned, evidenced", () => {
    for (const spec of FIXTURES) {
      const trace = generateTraceFixture(spec);
      const input = trace.events[0] as Extract<
        (typeof trace.events)[number],
        { type: "INPUT" }
      >;
      const concepts = trace.events.filter(
        (e): e is Extract<typeof e, { type: "CONCEPT" }> => e.type === "CONCEPT",
      );
      expect(concepts.length).toBeGreaterThan(0);

      const byPosition = new Map<number, number[]>();
      for (const e of concepts) {
        // dictionary concepts only — no hand-placed labels survive
        expect(e.conceptId).toMatch(/^concept_[a-z_]+$/);
        expect(e.level).toBe("INTERPRETED");
        // threshold applied pre-rounding: 4-dp rounding slack only
        expect(e.score).toBeGreaterThanOrEqual(CONCEPT_ACTIVE_MASS - 5e-5);
        expect(e.positions).toHaveLength(1);
        const position = e.positions![0];
        expect(position).toBeGreaterThanOrEqual(input.tokenCount);
        expect(position).toBeLessThan(input.tokenCount + trace.output!.tokenCount);

        expect(e.evidence!.length).toBeGreaterThan(0);
        const concept = CONCEPT_DICTIONARY.find((c) => c.conceptId === e.conceptId)!;
        for (const ev of e.evidence!) {
          expect(concept.words).toContain(ev.text); // evidence stays auditable
          expect(ev.probability).toBeGreaterThan(0);
        }
        const probs = e.evidence!.map((x) => x.probability);
        expect(probs).toEqual([...probs].sort((a, b) => b - a));
        const scores = byPosition.get(position) ?? [];
        scores.push(e.score);
        byPosition.set(position, scores);
      }
      // within a step, events arrive sorted by mass (descending)
      for (const scores of byPosition.values()) {
        expect(scores).toEqual([...scores].sort((a, b) => b - a));
      }
    }
  });

  it("RESEARCH fixtures carry concepts alongside attention", () => {
    const skyBlue = FIXTURES.find((f) => f.key === "trace-sky-blue")!;
    const trace = generateTraceFixture(skyBlue);
    const types = trace.events.map((e) => e.type);
    expect(types).toContain("ATTENTION");
    expect(types).toContain("CONCEPT");
    // every step: TOKEN, LAYER_ACTIVITY, 12×ATTENTION, then CONCEPT×k (k ≥ 0)
    const tokenIdx = [...types.keys()].filter((i) => types[i] === "TOKEN");
    for (let s = 0; s < tokenIdx.length - 1; s++) {
      const block = types.slice(tokenIdx[s], tokenIdx[s + 1]);
      expect(block[0]).toBe("TOKEN");
      expect(block[1]).toBe("LAYER_ACTIVITY");
      expect(block.slice(2, 14)).toEqual(Array.from({ length: 12 }, () => "ATTENTION"));
      expect(block.slice(14).every((t) => t === "CONCEPT")).toBe(true);
    }
  });
});

describe("schema strictness", () => {
  it("rejects unknown fields", () => {
    const trace = generateTraceFixture(FIXTURES[2]) as unknown as Record<string, unknown>;
    trace.surprise = true;
    expect(() => traceSchema.parse(trace)).toThrow();
  });

  it("rejects malformed event ids", () => {
    const trace = generateTraceFixture(FIXTURES[2]);
    const bad = JSON.parse(JSON.stringify(trace));
    bad.events[1].id = "nope";
    expect(() => traceSchema.parse(bad)).toThrow();
  });
});

describe("concept dictionary (Phase 5)", () => {
  it("word sets are disjoint — mass attributes to exactly one label", () => {
    const seen = new Map<string, string>();
    for (const concept of CONCEPT_DICTIONARY) {
      for (const word of concept.words) {
        const owner = seen.get(word);
        if (owner) {
          throw new Error(`"${word}" is in both ${owner} and ${concept.conceptId}`);
        }
        seen.set(word, concept.conceptId);
      }
    }
    expect(seen.size).toBeGreaterThan(50);
  });

  it("is a real dictionary: ≥10 concepts, ≥8 single words each, sane threshold", () => {
    expect(CONCEPT_DICTIONARY.length).toBeGreaterThanOrEqual(10);
    for (const concept of CONCEPT_DICTIONARY) {
      expect(concept.words.length).toBeGreaterThanOrEqual(8);
      for (const word of concept.words) {
        expect(word).toMatch(/^[a-z]+$/); // single lowercase words, no phrases
      }
      expect(concept.conceptId).toMatch(/^concept_[a-z_]+$/);
    }
    expect(CONCEPT_ACTIVE_MASS).toBeGreaterThan(0);
    expect(CONCEPT_ACTIVE_MASS).toBeLessThan(1);
  });

  it("CONCEPT events accept auditable evidence", () => {
    const event = {
      id: "evt_0001",
      seq: 1,
      type: "CONCEPT",
      t: 10,
      level: "INTERPRETED",
      conceptId: "concept_uncertainty",
      label: "uncertainty / hedging",
      score: 0.42,
      positions: [7],
      evidence: [
        { tokenId: 1, text: "maybe", probability: 0.4 },
        { tokenId: 2, text: "perhaps", probability: 0.02 },
      ],
    };
    expect(traceSchema.parse({ ...baseTrace(), events: [event] }).events[0])
      .toMatchObject({ conceptId: "concept_uncertainty", score: 0.42 });
    // evidence is optional (old fixtures carry none) but bounded
    const noEvidence = { ...event } as Record<string, unknown>;
    delete noEvidence.evidence;
    expect(() => traceSchema.parse({ ...baseTrace(), events: [noEvidence] })).not.toThrow();
    const tooMany = {
      ...event,
      evidence: Array.from({ length: 9 }, () => ({ tokenId: 1, text: "x", probability: 0.1 })),
    };
    expect(() => traceSchema.parse({ ...baseTrace(), events: [tooMany] })).toThrow();
  });
});

describe("signal taxonomy", () => {
  it("covers every rendered Inspector field key", () => {
    const required = [
      "token.probability",
      "token.rank",
      "token.entropyBits",
      "token.latencyMs",
      "token.topK",
      "layer.l2Norm",
      "layer.normRatio",
      "concept.score",
      "decision.reason",
      "output.durationMs",
      "input.tokens",
    ];
    for (const key of required) {
      expect(SIGNAL_TAXONOMY[key], `missing taxonomy entry: ${key}`).toBeDefined();
      expect(SIGNAL_TAXONOMY[key].definition.length).toBeGreaterThan(10);
    }
  });
});
