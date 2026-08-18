/**
 * Trace store — the single state every data-source mode feeds.
 *
 * `applyEvent` is a pure function (unit-tested without React): fixture
 * replay, live SSE, and saved playback all push events through it, so the
 * explorer behaves identically regardless of where a trace came from.
 */
import { create } from "zustand";
import type {
  AttentionEvent,
  ConceptEvent,
  Trace,
  TraceEvent,
  TokenEvent,
  LayerActivityEvent,
  OutputEvent,
} from "@ets/trace-schema";

export type DataSourceMode = "fixture" | "live" | "replay";
export type StreamStatus = "idle" | "streaming" | "complete" | "error";

export interface TraceState {
  mode: DataSourceMode;
  envelope: Trace | null;
  events: TraceEvent[];
  status: StreamStatus;
  error: string | null;
  /** selected event id (evt_NNNN) — drives the Inspector + canvas ring */
  selectedEventId: string | null;
}

export const initialTraceState: TraceState = {
  mode: "fixture",
  envelope: null,
  events: [],
  status: "idle",
  error: null,
  selectedEventId: null,
};

/** Pure: apply one validated trace event to the state. */
export function applyEvent(state: TraceState, event: TraceEvent): TraceState {
  // seq is gapless and totally orders a trace — a duplicate or regression
  // is a contract violation, ignore it rather than corrupt ordering.
  const lastSeq = state.events.at(-1)?.seq ?? -1;
  if (event.seq <= lastSeq) return state;

  return {
    ...state,
    events: [...state.events, event],
    selectedEventId:
      state.selectedEventId === null ? event.id : state.selectedEventId,
  };
}

/** Pure: mark the stream finished (terminal `done` frame / end of replay). */
export function completeStream(state: TraceState): TraceState {
  if (state.status === "complete") return state;
  return { ...state, status: "complete" };
}

/** Pure: mark the stream failed — unless it already completed. */
export function failStream(state: TraceState, message: string): TraceState {
  if (state.status !== "streaming") return state;
  return { ...state, status: "error", error: message };
}

// ——— selectors (pure) ——————————————————————————————————

export function tokenEvents(events: TraceEvent[]): TokenEvent[] {
  return events.filter((e): e is TokenEvent => e.type === "TOKEN");
}

export function outputEvent(events: TraceEvent[]): OutputEvent | null {
  return (
    (events.find((e): e is OutputEvent => e.type === "OUTPUT") ?? null)
  );
}

export function layerActivityByPosition(
  events: TraceEvent[],
): Map<number, LayerActivityEvent> {
  const map = new Map<number, LayerActivityEvent>();
  for (const e of events) {
    if (e.type === "LAYER_ACTIVITY") map.set(e.position, e);
  }
  return map;
}

/**
 * ATTENTION events for each position, in layer order (the engine emits
 * L01…L12 back to back after the TOKEN event). RESEARCH traces only.
 */
export function attentionByPosition(
  events: TraceEvent[],
): Map<number, AttentionEvent[]> {
  const map = new Map<number, AttentionEvent[]>();
  for (const e of events) {
    if (e.type !== "ATTENTION") continue;
    const list = map.get(e.position);
    if (list) list.push(e);
    else map.set(e.position, [e]);
  }
  return map;
}

/**
 * CONCEPT events carry no position of their own — they attach to the most
 * recent TOKEN event at their point in the seq order.
 */
export function conceptsByToken(
  events: TraceEvent[],
): Map<string, ConceptEvent[]> {
  const map = new Map<string, ConceptEvent[]>();
  let currentToken: TokenEvent | null = null;
  for (const e of events) {
    if (e.type === "TOKEN") currentToken = e;
    else if (e.type === "CONCEPT" && currentToken) {
      const list = map.get(currentToken.id) ?? [];
      list.push(e);
      map.set(currentToken.id, list);
    }
  }
  return map;
}

export function selectedEvent(state: TraceState): TraceEvent | null {
  return state.events.find((e) => e.id === state.selectedEventId) ?? null;
}

// ——— zustand binding —————————————————————————————————

interface TraceStore extends TraceState {
  begin(mode: DataSourceMode, envelope: Trace): void;
  accept(event: TraceEvent): void;
  complete(): void;
  fail(message: string): void;
  select(eventId: string | null): void;
  reset(): void;
}

export const useTraceStore = create<TraceStore>((set) => ({
  ...initialTraceState,
  begin: (mode, envelope) =>
    set({ ...initialTraceState, mode, envelope, status: "streaming" }),
  accept: (event) => set((s) => applyEvent(s, event)),
  complete: () => set((s) => completeStream(s)),
  fail: (message) => set((s) => failStream(s, message)),
  select: (eventId) => set({ selectedEventId: eventId }),
  reset: () => set(initialTraceState),
}));
