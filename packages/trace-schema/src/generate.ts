/**
 * Deterministic fixture generator (spec Phase 1).
 *
 * Produces Trace JSON shaped exactly like the real engine's output so the UI
 * is designed against honest signal shapes, not lies:
 *   - entropy is low on function words, spikes at content-word choice points
 *   - the emitted token's probability tracks 2^-H (greedy decoding)
 *   - top-k probabilities descend geometrically, respecting the step's entropy
 *   - residual l2Norm grows with depth (real residual streams do)
 *   - t advances by plausible per-step latencies
 *   - CONCEPT events are scored from the same fake distribution the
 *     engine's FakeBackend builds (Phase 5): dictionary mass on the
 *     renormalized top-k + bump distribution, evidence attached, so the
 *     fixtures exercise the exact INTERPRETED-event shape the engine emits
 *
 * Same seed ⇒ byte-identical JSON (mulberry32; no Date.now, no Math.random).
 * The realism math is deliberately portable: the Python FakeBackend mirrors it.
 */
import type {
  AttentionEvent,
  ConceptEvent,
  ConceptEvidence,
  DecisionEvent,
  InputEvent,
  LayerActivityEvent,
  LayerStat,
  OutputEvent,
  TokenEvent,
  TopToken,
  Trace,
  TraceEvent,
} from "./events";
import {
  CONCEPT_ACTIVE_MASS,
  CONCEPT_DICTIONARY,
  CONCEPT_EVIDENCE_LIMIT,
} from "./concepts";

export interface FixtureSpec {
  key: string;
  displayId: number;
  prompt: string;
  continuation: string;
  seed: number;
  layerCount: number;
  /** Emit per-layer ATTENTION events (fixture becomes RESEARCH mode). */
  attention?: boolean;
  headCount?: number;
}

/** Deterministic PRNG — identical implementation must live in fake_backend.py. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const FUNCTION_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
  "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "it",
  "its", "this", "that", "these", "those", "you", "your", "we", "our", "they",
  "their", "he", "she", "his", "her", "i", "if", "then", "than", "so", "not",
  "no", "can", "will", "would", "should", "may", "might", "do", "does", "did",
  "has", "have", "had", "into", "about", "over", "under", "more", "most",
  "some", "any", "each", "which", "who", "when", "where", "how", "what",
]);

/** Stable word → pseudo-token-id (GPT-2 vocab is 50257). */
function wordToTokenId(word: string): number {
  let h = 2166136261;
  for (let i = 0; i < word.length; i++) {
    h ^= word.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return 1000 + (h >>> 0) % 49000;
}

/** Split text into pseudo-BPE tokens: leading-space aware, punctuation attached. */
export function splitTokens(text: string): { text: string; leadingSpace: boolean }[] {
  const out: { text: string; leadingSpace: boolean }[] = [];
  for (const piece of text.split(" ")) {
    if (piece === "") continue;
    const leadingSpace = out.length > 0;
    // split trailing punctuation (",", ".", ";", ":", "?", "!") into its own token
    const m = piece.match(/^(.*?)([.,;:?!]+)$/);
    if (m && m[1] !== "") {
      out.push({ text: m[1], leadingSpace });
      out.push({ text: m[2], leadingSpace: false });
    } else {
      out.push({ text: piece, leadingSpace });
    }
  }
  return out;
}

function classify(text: string): "function" | "punct" | "content" {
  if (/^[.,;:?!]+$/.test(text)) return "punct";
  if (FUNCTION_WORDS.has(text.toLowerCase())) return "function";
  return "content";
}

function round(x: number, decimals: number): number {
  const f = 10 ** decimals;
  return Math.round(x * f) / f;
}

function seededShuffle<T>(items: T[], rng: () => number): T[] {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/** Residual norms grow with depth; per-step gain makes normRatio breathe. */
function layerNormCurve(layer: number, layerCount: number, stepGain: number): number {
  const frac = layer / layerCount;
  return (2.5 + 2.4 * layer + 6 * frac * frac) * stepGain;
}

/**
 * Deterministic imitation of the hook-side attention reduction — the exact
 * mirror of FakeBackend._fake_attention: BOS sink growing with depth,
 * recency-weighted jitter elsewhere, rows normalized to sum 1, head mean +
 * per-head entropy. Draws from a SEPARATE seeded stream (seed + 777_000 +
 * step*131 + layer*37) so adding attention never shifts the main rng —
 * STANDARD fixtures stay byte-identical to before this existed.
 */
export function fakeAttention(
  seed: number,
  step: number,
  layer: number, // 0-based
  layerCount: number,
  headCount: number,
  n: number, // context length INCLUDING the BOS entry
): { mean: number[]; headEntropyBits: number[] } {
  const rng = mulberry32(seed + 777_000 + step * 131 + layer * 37);
  const rows: number[][] = [];
  for (let h = 0; h < headCount; h++) {
    const w: number[] = new Array(n);
    const bosBias = 0.2 + (0.6 * (layer + 1)) / layerCount;
    let total = 0;
    for (let j = 0; j < n; j++) {
      w[j] =
        j === 0
          ? bosBias + rng() * 0.25
          : Math.exp(-(n - 1 - j) / 6) * (0.3 + rng() * 0.9);
      total += w[j];
    }
    rows.push(w.map((x) => x / total));
  }
  const mean = new Array<number>(n);
  for (let j = 0; j < n; j++) {
    mean[j] = rows.reduce((a, r) => a + r[j], 0) / headCount;
  }
  const headEntropyBits = rows.map((r) =>
    round(-r.reduce((a, p) => (p > 0 ? a + p * Math.log2(p) : a), 0), 3),
  );
  return { mean, headEntropyBits };
}

/**
 * Deterministic concept mass for one step's fake distribution — the exact
 * mirror of FakeBackend._concept_bump: 0/1/2 concepts per step, mass spread
 * over up to 3 of the concept's words, skipping ids that already carry
 * top-k mass. Draws from a SEPARATE seeded stream (seed + 888_000 + step*173)
 * so the main rng — and every pre-existing field — is untouched.
 */
export function fakeConceptBump(
  seed: number,
  step: number,
  occupied: Set<number>,
): Map<number, { word: string; share: number }> {
  const rng = mulberry32(seed + 888_000 + step * 173);
  const order = seededShuffle(
    CONCEPT_DICTIONARY.map((_, i) => i),
    rng,
  );
  const first = rng();
  const nActive = first < 0.1 ? 0 : rng() < 0.72 ? 1 : 2;

  const bump = new Map<number, { word: string; share: number }>();
  for (const idx of order.slice(0, nActive)) {
    const concept = CONCEPT_DICTIONARY[idx];
    const words = seededShuffle(concept.words, rng);
    const mass = 0.12 + rng() * 0.22;
    const picked: { tokenId: number; word: string }[] = [];
    for (const word of words) {
      if (picked.length >= 3) break;
      const tokenId = wordToTokenId(word);
      if (occupied.has(tokenId) || bump.has(tokenId)) continue;
      picked.push({ tokenId, word });
    }
    if (picked.length) {
      // concept mass == authored mass regardless of skips
      const share = mass / picked.length;
      for (const { tokenId, word } of picked) bump.set(tokenId, { word, share });
    }
  }
  return bump;
}

export function generateTraceFixture(spec: FixtureSpec): Trace {
  const rng = mulberry32(spec.seed);
  const events: TraceEvent[] = [];
  let seq = 0;
  let t = 3; // ms — trace start overhead

  const nextId = () => `evt_${String(seq).padStart(4, "0")}`;

  const promptTokens = splitTokens(spec.prompt);

  const inputEvent: InputEvent = {
    id: nextId(),
    seq: seq++,
    type: "INPUT",
    t,
    level: "MEASURED",
    text: spec.prompt,
    tokenCount: promptTokens.length,
    tokens: promptTokens.map((tok, i) => ({
      position: i,
      tokenId: wordToTokenId(tok.text),
      text: tok.text,
    })),
  };
  events.push(inputEvent);

  const continuationTokens = splitTokens(spec.continuation);
  const promptLen = promptTokens.length;

  // topical alternatives: sample from the continuation's own vocabulary
  const vocab = [
    ...new Set(
      continuationTokens
        .map((tok) => tok.text.toLowerCase())
        .filter((w) => w.length > 2 && !FUNCTION_WORDS.has(w)),
    ),
  ];
  const fallbackPool = [
    "approach", "context", "detail", "structure", "process", "signal",
    "pattern", "framework", "consider", "suggest", "provide", "reflect",
  ];

  // running per-layer mean for normRatio (includes current step; step 0 ⇒ 1.0)
  const layerMeans = new Array<number>(spec.layerCount).fill(0);

  // position-indexed texts for attention rows; entry 0 is the BOS every real
  // context starts with — surfaced as position -1, never dropped
  const ctxTexts = ["<bos>", ...promptTokens.map((tok) => tok.text)];

  const outputTokenEvents: TokenEvent[] = [];

  for (let step = 0; step < continuationTokens.length; step++) {
    const tok = continuationTokens[step];
    const kind = classify(tok.text);

    // entropy model: function/punct ≈ determined; content contested; ~18% spikes
    let entropyBits: number;
    if (kind === "punct") entropyBits = 0.05 + rng() * 0.3;
    else if (kind === "function") entropyBits = 0.1 + rng() * 0.8;
    else if (rng() < 0.18) entropyBits = 3.5 + rng() * 3.0;
    else entropyBits = 1.2 + rng() * 2.0;
    entropyBits = round(entropyBits, 4);

    // greedy decoding ⇒ emitted token is argmax; p tracks 2^-H with jitter
    const probability = round(
      Math.min(0.995, Math.max(0.02, 2 ** -entropyBits * (0.85 + rng() * 0.3))),
      4,
    );

    // geometric decay for alternatives; higher entropy ⇒ slower decay
    const decay = Math.min(0.85, Math.max(0.05, 2 ** (-1 / Math.max(entropyBits, 0.4))));
    const alternatives = seededShuffle(
      vocab.filter((w) => w !== tok.text.toLowerCase()).concat(fallbackPool),
      rng,
    );

    const topK: TopToken[] = [];
    const emitted: TopToken = {
      tokenId: wordToTokenId(tok.text),
      text: tok.text,
      rawText: (tok.leadingSpace ? "Ġ" : "") + tok.text,
      leadingSpace: tok.leadingSpace,
      probability,
      rank: 0,
    };
    topK.push(emitted);
    let p = probability;
    for (let i = 0; i < 7 && i < alternatives.length; i++) {
      p *= decay;
      const alt = alternatives[i];
      topK.push({
        tokenId: wordToTokenId(alt),
        text: alt,
        rawText: "Ġ" + alt,
        leadingSpace: true,
        probability: round(p, 4),
        rank: i + 1,
      });
    }

    const latencyMs = Math.round(38 + rng() * 45 + (step % 4 === 0 ? 25 : 0));
    t += latencyMs + 15; // forward pass + emit overhead

    const tokenEvent: TokenEvent = {
      id: nextId(),
      seq: seq++,
      type: "TOKEN",
      t,
      level: "MEASURED",
      position: promptLen + step,
      step,
      tokenId: emitted.tokenId,
      text: tok.text,
      rawText: emitted.rawText,
      leadingSpace: tok.leadingSpace,
      probability,
      rank: 0,
      entropyBits,
      topK,
      latencyMs,
    };
    events.push(tokenEvent);
    outputTokenEvents.push(tokenEvent);

    // layer activity (STANDARD mode fixtures)
    const stepGain = 0.9 + rng() * 0.25;
    const layers: LayerStat[] = [];
    for (let i = 0; i < spec.layerCount; i++) {
      const l2Norm = round(layerNormCurve(i + 1, spec.layerCount, stepGain), 3);
      // update running mean including this step (Welford not needed for a mean)
      layerMeans[i] = layerMeans[i] === 0 ? l2Norm : layerMeans[i] + (l2Norm - layerMeans[i]) / (step + 1);
      layers.push({ layer: i + 1, l2Norm, normRatio: round(l2Norm / layerMeans[i], 3) });
    }
    const layerEvent: LayerActivityEvent = {
      id: nextId(),
      seq: seq++,
      type: "LAYER_ACTIVITY",
      t,
      level: "DERIVED",
      position: promptLen + step,
      step,
      layers,
    };
    events.push(layerEvent);

    // per-layer attention rows from the final position (RESEARCH fixtures)
    if (spec.attention) {
      const headCount = spec.headCount ?? 12;
      for (let i = 0; i < spec.layerCount; i++) {
        const { mean, headEntropyBits } = fakeAttention(
          spec.seed,
          step,
          i,
          spec.layerCount,
          headCount,
          ctxTexts.length,
        );
        const attentionEvent: AttentionEvent = {
          id: nextId(),
          seq: seq++,
          type: "ATTENTION",
          t,
          level: "DERIVED",
          position: promptLen + step,
          layer: i + 1,
          aggregated: mean.map((weight, j) => ({
            position: j - 1,
            text: ctxTexts[j],
            weight: round(weight, 4),
          })),
          headEntropyBits,
        };
        events.push(attentionEvent);
      }
    }

    // concepts: exact mass the renormalized fake distribution places on
    // each dictionary word set — the same method as the engine's
    // ConceptScorer (mass MEASURED, label INTERPRETED, evidence shipped)
    const dist = new Map<number, number>();
    for (const cand of topK) {
      dist.set(cand.tokenId, Math.max(dist.get(cand.tokenId) ?? 0, cand.probability));
    }
    const bump = fakeConceptBump(spec.seed, step, new Set(dist.keys()));
    for (const [tokenId, { share }] of bump) {
      dist.set(tokenId, (dist.get(tokenId) ?? 0) + share);
    }
    const total = [...dist.values()].reduce((a, b) => a + b, 0);

    const active: {
      conceptId: string;
      label: string;
      mass: number;
      evidence: ConceptEvidence[];
    }[] = [];
    for (const concept of CONCEPT_DICTIONARY) {
      let mass = 0;
      const carrying: { tokenId: number; word: string; p: number }[] = [];
      for (const word of concept.words) {
        const raw = dist.get(wordToTokenId(word));
        if (!raw) continue;
        const p = raw / total;
        mass += p;
        if (p > 1e-4) carrying.push({ tokenId: wordToTokenId(word), word, p });
      }
      if (mass < CONCEPT_ACTIVE_MASS) continue;
      carrying.sort((a, b) => b.p - a.p);
      active.push({
        conceptId: concept.conceptId,
        label: concept.label,
        mass,
        evidence: carrying.slice(0, CONCEPT_EVIDENCE_LIMIT).map(({ tokenId, word, p }) => ({
          tokenId,
          text: word,
          probability: round(p, 4),
        })),
      });
    }
    active.sort((a, b) => b.mass - a.mass);
    for (const c of active) {
      const conceptEvent: ConceptEvent = {
        id: nextId(),
        seq: seq++,
        type: "CONCEPT",
        t,
        level: "INTERPRETED",
        conceptId: c.conceptId,
        label: c.label,
        score: round(c.mass, 4),
        positions: [promptLen + step],
        evidence: c.evidence,
      };
      events.push(conceptEvent);
    }

    ctxTexts.push(tok.text);
  }

  const decisionEvent: DecisionEvent = {
    id: nextId(),
    seq: seq++,
    type: "DECISION",
    t: t + 4,
    level: "DERIVED",
    decision: "stop",
    detail: "end-of-sequence token reached",
  };
  events.push(decisionEvent);

  const outputEvent: OutputEvent = {
    id: nextId(),
    seq: seq++,
    type: "OUTPUT",
    t: t + 8,
    level: "MEASURED",
    text: spec.continuation.trim(),
    tokenCount: continuationTokens.length,
    durationMs: t + 8,
    finishReason: "stop",
  };
  events.push(outputEvent);

  const traceId = `tr_${spec.seed.toString(36).padStart(8, "0")}`.slice(0, 11);

  return {
    id: traceId,
    displayId: spec.displayId,
    model: {
      name: "gpt2-small",
      revision: "fixture",
      device: "fixture",
      layerCount: spec.layerCount,
      paramCount: 124_000_000,
    },
    input: { text: spec.prompt },
    traceMode: spec.attention ? "RESEARCH" : "STANDARD",
    sampling: {
      maxTokens: continuationTokens.length + 2,
      temperature: 0,
      topK: null,
      seed: spec.seed,
    },
    status: "complete",
    // fixed timestamp: fixtures must be byte-stable across regenerations
    createdAt: "2026-08-16T00:00:00.000Z",
    output: {
      text: spec.continuation.trim(),
      tokenCount: continuationTokens.length,
      durationMs: outputEvent.durationMs,
      finishReason: "stop",
    },
    events,
  };
}

/** The committed fixture corpus (spec §31 demo prompt + sky blue + minimal). */
export const FIXTURES: FixtureSpec[] = [
  {
    key: "trace-python-rust",
    displayId: 142,
    prompt: "Should I learn Python or Rust?",
    seed: 4242,
    layerCount: 12,
    continuation:
      "For most people starting out, Python is the better first language. " +
      "The ecosystem is enormous, the syntax reads almost like English, and " +
      "you can move from small scripts to data work to web services without " +
      "switching tools. Rust asks more of you up front: ownership, borrowing, " +
      "and lifetimes are real concepts you must understand before the " +
      "compiler becomes your friend rather than your adversary. That rigor " +
      "pays off in systems programming, where control over memory and " +
      "performance matters. A reasonable path is Python first, build real " +
      "things, then learn Rust when a project genuinely demands it. Learning " +
      "Rust without a motivating problem tends to end in frustration.",
  },
  {
    key: "trace-sky-blue",
    displayId: 87,
    prompt: "Why is the sky blue?",
    seed: 777,
    layerCount: 12,
    attention: true, // the RESEARCH-mode fixture: per-layer ATTENTION events
    headCount: 12,
    continuation:
      "The sky looks blue because of how air molecules scatter sunlight. " +
      "Sunlight contains all colors, and short blue wavelengths scatter more " +
      "strongly than long red ones when they strike particles in the " +
      "atmosphere. This effect is called Rayleigh scattering. The scattered " +
      "blue light reaches your eyes from every direction, so the whole sky " +
      "glows blue. At sunset light travels a longer path, most blue is " +
      "scattered away, and the sky turns red and orange.",
  },
  {
    key: "trace-minimal",
    displayId: 1,
    prompt: "Hi",
    seed: 1,
    layerCount: 12,
    continuation: "Hello there .",
  },
];
