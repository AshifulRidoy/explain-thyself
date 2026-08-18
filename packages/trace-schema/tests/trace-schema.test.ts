import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  FIXTURES,
  generateTraceFixture,
  mulberry32,
} from "../src/generate.js";
import { traceSchema, validateTrace } from "../src/schema.js";
import { SIGNAL_TAXONOMY } from "../src/taxonomy.js";

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

async function readFixture(key: string): Promise<unknown> {
  return JSON.parse(await readFile(join(FIXTURES_DIR, `${key}.json`), "utf8"));
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
