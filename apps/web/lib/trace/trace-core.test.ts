/**
 * Pure-core tests: the reducer, the graph builder, and the SSE parser —
 * the three pieces every data-source mode depends on.
 */
import { describe, expect, it } from "vitest";
import { loadFixture } from "@ets/trace-schema/fixtures";
import {
  applyEvent,
  attentionByPosition,
  completeStream,
  conceptsByPosition,
  conceptsByToken,
  conceptsTimeline,
  failStream,
  initialTraceState,
  layerActivityByPosition,
  tokenEvents,
} from "./store";
import {
  uncertaintyQuantities,
  uncertaintyTaxonomyKey,
  UNCERTAINTY_LABELS,
} from "./uncertainty";
import { toGraph } from "./graph";
import { SseParser } from "./sse";

const pyRust = await loadFixture("trace-python-rust");
const skyBlue = await loadFixture("trace-sky-blue");

describe("applyEvent", () => {
  it("appends events in order and auto-selects the first", () => {
    let state = initialTraceState;
    for (const e of pyRust.events.slice(0, 10)) {
      state = applyEvent(state, e);
    }
    expect(state.events).toHaveLength(10);
    expect(state.selectedEventId).toBe(pyRust.events[0].id);
  });

  it("ignores duplicate or regressing seq — ordering can never corrupt", () => {
    let state = initialTraceState;
    state = applyEvent(state, pyRust.events[0]);
    state = applyEvent(state, pyRust.events[0]); // duplicate
    state = applyEvent(state, pyRust.events[0]); // regressed
    expect(state.events).toHaveLength(1);
  });

  it("terminal transitions are one-way", () => {
    let state = completeStream(initialTraceState);
    state = failStream(state, "x");
    expect(state.status).toBe("complete");
  });
});

describe("selectors", () => {
  it("tokenEvents extracts only TOKEN events in order", () => {
    const tokens = tokenEvents(pyRust.events);
    expect(tokens.length).toBeGreaterThan(10);
    expect(tokens[0].position).toBeLessThan(tokens[1].position);
  });

  it("conceptsByToken attaches CONCEPTs to the preceding token", () => {
    const map = conceptsByToken(pyRust.events);
    const all = [...map.values()].flat();
    expect(all.length).toBeGreaterThan(0);
    // every attached concept's token id must exist in the token list
    const tokenIds = new Set(tokenEvents(pyRust.events).map((t) => t.id));
    for (const id of map.keys()) expect(tokenIds.has(id)).toBe(true);
  });

  it("conceptsTimeline ranks concepts by total mass, peak = max score", () => {
    const timeline = conceptsTimeline(pyRust.events);
    expect(timeline.length).toBeGreaterThan(3);
    // ranking: totalMass descending
    const totals = timeline.map((a) => a.totalMass);
    expect(totals).toEqual([...totals].sort((a, b) => b - a));
    for (const activity of timeline) {
      expect(activity.totalMass).toBe(
        activity.events.reduce((sum, e) => sum + e.score, 0),
      );
      const peakScore = Math.max(...activity.events.map((e) => e.score));
      expect(activity.peak.score).toBe(peakScore);
      expect(activity.peak.conceptId).toBe(activity.conceptId);
    }
    // aggregates stay consistent with the raw events
    const allEvents = timeline.flatMap((a) => a.events);
    expect(allEvents.length).toBe(
      pyRust.events.filter((e) => e.type === "CONCEPT").length,
    );
  });

  it("conceptsByPosition keys events by their measured position", () => {
    const byPos = conceptsByPosition(pyRust.events);
    const input = pyRust.events[0] as { tokenCount: number };
    for (const [position, events] of byPos) {
      expect(position).toBeGreaterThanOrEqual(input.tokenCount);
      for (const e of events) expect(e.positions).toContain(position);
    }
    // partial streams: only landed positions appear (streaming aggregation)
    const partial = conceptsByPosition(pyRust.events.slice(0, 20));
    for (const events of partial.values()) {
      for (const e of events) expect(pyRust.events.slice(0, 20)).toContain(e);
    }
  });

  it("layerActivityByPosition maps every STANDARD token position", () => {
    const map = layerActivityByPosition(pyRust.events);
    for (const tok of tokenEvents(pyRust.events)) {
      expect(map.get(tok.position)?.position).toBe(tok.position);
    }
  });
});

describe("attentionByPosition (RESEARCH traces)", () => {
  it("groups per-layer events by position, layer-ascending", () => {
    const map = attentionByPosition(skyBlue.events);
    const tokens = tokenEvents(skyBlue.events);
    expect(map.size).toBe(tokens.length);
    for (const tok of tokens) {
      const layers = map.get(tok.position)!;
      expect(layers).toHaveLength(12);
      expect(layers.map((a) => a.layer)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
      for (const a of layers) expect(a.position).toBe(tok.position);
    }
  });

  it("keeps partial streams usable — events apply incrementally", () => {
    // mid-stream: INPUT + first TOKEN + its LAYER_ACTIVITY + 5 of 12 layers
    const firstAttn = skyBlue.events.findIndex((e) => e.type === "ATTENTION");
    let state = initialTraceState;
    for (const e of skyBlue.events.slice(0, firstAttn + 5)) {
      state = applyEvent(state, e);
    }
    const map = attentionByPosition(state.events);
    expect(map.size).toBe(1);
    expect(map.get(tokenEvents(state.events)[0].position)).toHaveLength(5);
  });

  it("returns an empty map for STANDARD traces", () => {
    expect(attentionByPosition(pyRust.events).size).toBe(0);
  });
});

describe("uncertaintyQuantities (spec §22)", () => {
  it("returns the four kinds in spec order for RESEARCH, none for STANDARD", () => {
    const quantities = uncertaintyQuantities(skyBlue.events);
    expect(quantities.map((q) => q.kind)).toEqual([
      "MODEL_UNCERTAINTY",
      "EVIDENCE_QUALITY",
      "INPUT_AMBIGUITY",
      "ANSWER_STABILITY",
    ]);
    expect(uncertaintyQuantities(pyRust.events)).toEqual([]);
  });

  it("model uncertainty is auditable from the trace's own TOKEN events", () => {
    const quantities = uncertaintyQuantities(skyBlue.events);
    const tokens = tokenEvents(skyBlue.events);
    const mean = tokens.reduce((a, e) => a + e.entropyBits, 0) / tokens.length;
    expect(quantities[0].value).toBeCloseTo(mean / Math.log2(50_000), 4);

    // stability = mean agreement over the shipped variants, recomputable
    const stability = quantities[3];
    const agreed = stability.variants!.reduce((a, v) => a + v.agreedTokens, 0);
    const total = stability.variants!.reduce((a, v) => a + v.totalTokens, 0);
    expect(stability.value).toBeCloseTo(agreed / total, 4);

    // the two nulls carry their refusal in the event itself
    for (const skipped of [quantities[1], quantities[2]]) {
      expect(skipped.value).toBeNull();
      expect(skipped.level).toBeNull();
      expect(skipped.basis).toMatch(/^not measured/);
    }
  });

  it("labels and taxonomy keys cover every kind", () => {
    for (const kind of [
      "MODEL_UNCERTAINTY",
      "EVIDENCE_QUALITY",
      "INPUT_AMBIGUITY",
      "ANSWER_STABILITY",
    ] as const) {
      expect(UNCERTAINTY_LABELS[kind]).toMatch(/^[a-z ]+$/);
      expect(uncertaintyTaxonomyKey(kind)).toBe(
        kind.toLowerCase().replace(/_([a-z])/g, (_, c: string) => c.toUpperCase()),
      );
    }
  });
});

describe("toGraph", () => {
  it("builds INPUT → token chain → OUTPUT with concept attachments", () => {
    const { nodes, edges } = toGraph(pyRust.events, null, false);
    const types = nodes.map((n) => n.type);
    expect(types[0]).toBe("INPUT");
    expect(types.at(-1)).toBe("OUTPUT");
    expect(types.filter((t) => t === "TOKEN")).toHaveLength(
      pyRust.events.filter((e) => e.type === "TOKEN").length,
    );
    expect(types).toContain("CONCEPT");

    // chain connectivity: input → t0 → t1 … → output, gapless
    const out = nodes.find((n) => n.type === "OUTPUT")!;
    const into: string[] = edges.filter((e) => e.target === out.id).map((e) => e.source);
    expect(into).toHaveLength(1);
    const tokenNodes = nodes.filter((n) => n.type === "TOKEN");
    expect(into[0]).toBe(tokenNodes.at(-1)!.id);
  });

  it("marks exactly the selected node", () => {
    const target = pyRust.events.find((e) => e.type === "TOKEN")!;
    const { nodes } = toGraph(pyRust.events, target.id, false);
    const selected = nodes.filter((n) => (n.data as { selected: boolean }).selected);
    expect(selected).toHaveLength(1);
    expect(selected[0].id).toBe(target.id);
  });

  it("partial traces (still streaming) end at the last applied token", () => {
    const partial = pyRust.events.slice(0, 6);
    const { nodes } = toGraph(partial, null, true);
    expect(nodes.at(-1)!.type).not.toBe("OUTPUT");
  });

  it("shows one ConceptNode per conceptId, at its peak-score step", () => {
    const conceptEvents = pyRust.events.filter((e) => e.type === "CONCEPT");
    expect(conceptEvents.length).toBeGreaterThan(30); // the trace is concept-rich
    const { nodes } = toGraph(pyRust.events, null, false);
    const conceptNodes = nodes.filter((n) => n.type === "CONCEPT");
    const distinctIds = new Set(conceptEvents.map((e) => e.conceptId));
    expect(conceptNodes).toHaveLength(distinctIds.size);

    // each node carries the peak (max-score) event for its concept, and a
    // concept peaks exactly once — no duplicate nodes can share an id
    const nodeIds = conceptNodes.map((n) => n.id);
    expect(new Set(nodeIds).size).toBe(nodeIds.length);
    for (const node of conceptNodes) {
      const event = (node.data as { event: { conceptId: string; score: number } }).event;
      const peak = Math.max(
        ...conceptEvents
          .filter((e) => e.conceptId === event.conceptId)
          .map((e) => e.score),
      );
      expect(event.score).toBe(peak);
    }
  });

  it("hangs UncertaintyNodes off the OUTPUT they analyze (RESEARCH)", () => {
    const uncertainty = uncertaintyQuantities(skyBlue.events);
    expect(uncertainty).toHaveLength(4);
    const { nodes, edges } = toGraph(skyBlue.events, null, false);
    const output = nodes.find((n) => n.type === "OUTPUT")!;

    const uNodes = nodes.filter((n) => n.type === "UNCERTAINTY");
    expect(uNodes).toHaveLength(4);
    // one node per event, id = event id (clicks select the real event)
    expect(uNodes.map((n) => n.id)).toEqual(uncertainty.map((e) => e.id));
    for (const node of uNodes) {
      const into = edges.filter((e) => e.target === node.id);
      expect(into).toHaveLength(1);
      expect(into[0].source).toBe(output.id);
    }

    // STANDARD traces stay clean — the layer is RESEARCH-only on the canvas
    const standard = toGraph(pyRust.events, null, false);
    expect(standard.nodes.filter((n) => n.type === "UNCERTAINTY")).toHaveLength(0);
  });
});

describe("SseParser", () => {
  it("parses frames from arbitrary chunk splits", () => {
    const parser = new SseParser();
    const stream =
      'event: trace\ndata: {"id":"tr_1"}\n\n: ping\n\nevent: trace_event\ndata: {"seq":0}\n\n';
    // split mid-line, mid-frame — boundaries are arbitrary
    const a = parser.push(stream.slice(0, 20));
    const b = parser.push(stream.slice(20, 37));
    const c = parser.push(stream.slice(37));
    const frames = [...a, ...b, ...c];
    expect(frames.map((f) => f.event)).toEqual(["trace", "trace_event"]);
    expect(frames[0].data).toBe('{"id":"tr_1"}');
  });

  it("skips heartbeat comments and handles CRLF framing", () => {
    const parser = new SseParser();
    const frames = parser.push("event: done\r\ndata: {}\r\n\r\n");
    expect(frames).toHaveLength(1);
    expect(frames[0].data).toBe("{}");
  });

  it("flush returns a trailing frame that never saw a blank line", () => {
    const parser = new SseParser();
    expect(parser.push("event: done\ndata: {}")).toHaveLength(0);
    expect(parser.flush()).toHaveLength(1);
  });
});
