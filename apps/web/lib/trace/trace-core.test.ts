/**
 * Pure-core tests: the reducer, the graph builder, and the SSE parser —
 * the three pieces every data-source mode depends on.
 */
import { describe, expect, it } from "vitest";
import { loadFixture } from "@ets/trace-schema/fixtures";
import {
  applyEvent,
  completeStream,
  conceptsByToken,
  failStream,
  initialTraceState,
  tokenEvents,
} from "./store";
import { toGraph } from "./graph";
import { SseParser } from "./sse";

const pyRust = await loadFixture("trace-python-rust");

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
