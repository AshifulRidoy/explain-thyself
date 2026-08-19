/**
 * state → React Flow graph. Pure, unit-tested.
 *
 * MVP topology (spec §15): INPUT → token₀ → token₁ → … → OUTPUT, with
 * ConceptNodes attached at their token. Deterministic hand-rolled wrap
 * layout — no dagre/elk, no animation of positions.
 */
import type { Edge, Node } from "@xyflow/react";
import type {
  ConceptEvent,
  LayerActivityEvent,
  TokenEvent,
  TraceEvent,
} from "@ets/trace-schema";

export interface TokenNodeData
  extends Record<string, unknown> {
  event: TokenEvent;
  layers: LayerActivityEvent | null;
  concepts: ConceptEvent[];
  selected: boolean;
  latest: boolean;
}

export interface GenericNodeData extends Record<string, unknown> {
  event: TraceEvent;
  selected: boolean;
  latest: boolean;
}

export type EtsNodeData = TokenNodeData | GenericNodeData;

const PER_ROW = 8;
const DX = 168;
const DY = 120;
const CONCEPT_OFFSET = { x: 60, y: 62 };

/** Pure: build React Flow nodes + edges from applied events. */
export function toGraph(
  events: TraceEvent[],
  selectedEventId: string | null,
  isStreaming: boolean,
): { nodes: Node<EtsNodeData>[]; edges: Edge[] } {
  const nodes: Node<EtsNodeData>[] = [];
  const edges: Edge[] = [];

  const input = events.find((e) => e.type === "INPUT");
  const output = events.find((e) => e.type === "OUTPUT");
  const tokens = events.filter((e): e is TokenEvent => e.type === "TOKEN");
  const layersByPos = new Map<number, LayerActivityEvent>();
  for (const e of events) {
    if (e.type === "LAYER_ACTIVITY") layersByPos.set(e.position, e);
  }
  // concepts attach to the most recent token at their point in seq order
  const conceptsByToken = new Map<string, ConceptEvent[]>();
  let currentToken: TokenEvent | null = null;
  for (const e of events) {
    if (e.type === "TOKEN") currentToken = e;
    else if (e.type === "CONCEPT" && currentToken) {
      const list = conceptsByToken.get(currentToken.id) ?? [];
      list.push(e);
      conceptsByToken.set(currentToken.id, list);
    }
  }

  // the canvas shows each concept ONCE per trace — at the step where it
  // peaked. A RESEARCH trace references a concept dozens of times; one node
  // per event would bury the token chain under hundreds of duplicates.
  const peakByConcept = new Map<string, ConceptEvent>();
  for (const e of events) {
    if (e.type !== "CONCEPT") continue;
    const best = peakByConcept.get(e.conceptId);
    if (!best || e.score > best.score) peakByConcept.set(e.conceptId, e);
  }
  const peaksByToken = new Map<string, ConceptEvent[]>();
  for (const tok of tokens) {
    const mine = (conceptsByToken.get(tok.id) ?? []).filter(
      (c) => peakByConcept.get(c.conceptId) === c,
    );
    if (mine.length) {
      mine.sort((a, b) => b.score - a.score);
      peaksByToken.set(tok.id, mine);
    }
  }

  let prevId: string | null = null;

  if (input) {
    nodes.push({
      id: input.id,
      type: "INPUT",
      position: { x: 0, y: 0 },
      data: { event: input, selected: input.id === selectedEventId, latest: false },
      draggable: false,
    });
    prevId = input.id;
  }

  tokens.forEach((tok, i) => {
    const col = i % PER_ROW;
    const row = Math.floor(i / PER_ROW);
    const id = tok.id;
    nodes.push({
      id,
      type: "TOKEN",
      position: { x: col * DX, y: (row + 1) * DY },
      data: {
        event: tok,
        layers: layersByPos.get(tok.position) ?? null,
        concepts: conceptsByToken.get(tok.id) ?? [],
        selected: id === selectedEventId,
        latest: i === tokens.length - 1 && isStreaming,
      },
      draggable: false,
    });
    if (prevId) {
      edges.push({
        id: `${prevId}->${id}`,
        source: prevId,
        target: id,
        style: { stroke: "var(--line)" },
      });
    }
    prevId = id;

    // concept peaks hang below their token, highest score first
    const concepts = peaksByToken.get(tok.id) ?? [];
    concepts.forEach((c, ci) => {
      // node id = the peak event's id (unique: one node per concept) so
      // canvas clicks select the event the Inspector will show
      nodes.push({
        id: c.id,
        type: "CONCEPT",
        position: {
          x: col * DX + CONCEPT_OFFSET.x,
          y: (row + 1) * DY + CONCEPT_OFFSET.y + ci * 56,
        },
        data: { event: c, selected: c.id === selectedEventId, latest: false },
        draggable: false,
      });
      edges.push({
        id: `${id}->${c.id}`,
        source: id,
        target: c.id,
        style: { stroke: "var(--line)", strokeDasharray: "2 3" },
      });
    });
  });

  if (output) {
    const i = tokens.length;
    const col = i % PER_ROW;
    const row = Math.floor(i / PER_ROW);
    nodes.push({
      id: output.id,
      type: "OUTPUT",
      position: { x: col * DX, y: (row + 1) * DY },
      data: {
        event: output,
        selected: output.id === selectedEventId,
        latest: false,
      },
      draggable: false,
    });
    if (prevId) {
      edges.push({
        id: `${prevId}->${output.id}`,
        source: prevId,
        target: output.id,
        style: { stroke: "var(--ink)" },
      });
    }
  }

  return { nodes, edges };
}
