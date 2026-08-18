"use client";

/**
 * THE swap seam (spec §49). One hook, three data sources:
 *
 *   fixture — a committed Trace replayed on its recorded `t` cadence
 *   live    — fetch-streamed SSE from the trace engine (POST :8000)
 *   replay  — a saved Trace played back through a rAF clock (M6)
 *
 * All three push events through the same `applyEvent` reducer, so Phase 1 →
 * Phase 2 was literally changing this file's mode argument — nothing else
 * in the explorer knows where a trace came from.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { traceEventSchema, type Trace, type TraceEvent } from "@ets/trace-schema";
import { useTraceStore, type DataSourceMode } from "./store";
import { SseParser } from "./sse";

export type PlaybackSpeed = 1 | 4 | 20;

export interface DataSourceControls {
  playing: boolean;
  speed: PlaybackSpeed;
  play: () => void;
  pause: () => void;
  reset: () => void;
  setSpeed: (s: PlaybackSpeed) => void;
}

const ENGINE_URL =
  process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";

// ——— fixture / replay: timed replay ———————————————————
// Any complete Trace — committed fixture or one loaded back from Postgres
// — plays on its recorded `t` cadence through the same reducer.

export function useFixtureReplay(
  trace: Trace,
  mode: DataSourceMode = "fixture",
): DataSourceControls {
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState<PlaybackSpeed>(4);
  const begin = useTraceStore((s) => s.begin);
  const accept = useTraceStore((s) => s.accept);
  const complete = useTraceStore((s) => s.complete);
  const resetStore = useTraceStore((s) => s.reset);

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const stateRef = useRef({ playing: true, speed: 4 as PlaybackSpeed });
  // recursive step through the ref: the scheduled closure must call the
  // CURRENT runFrom, not capture a stale one
  const runFromRef = useRef<(index: number) => void>(() => {});

  const stopTimer = () => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
  };

  const runFrom = useCallback(
    (index: number) => {
      stopTimer();
      const events = trace.events;
      if (index === 0) {
        resetStore();
        begin(mode, { ...trace, events: [] });
      }
      if (index >= events.length) {
        complete();
        setPlaying(false);
        return;
      }
      const prevT = index === 0 ? 0 : events[index - 1].t;
      const delay = Math.max(0, events[index].t - prevT);
      const { playing: isPlaying, speed: spd } = stateRef.current;
      if (!isPlaying) return;
      timer.current = setTimeout(() => {
        accept(events[index]);
        runFromRef.current(index + 1);
      }, delay / spd);
    },
    [trace, mode, accept, begin, complete, resetStore],
  );

  useEffect(() => {
    runFromRef.current = runFrom;
  }, [runFrom]);

  // the single scheduler. Callers mount this hook with key={trace.id}, so
  // a mount IS a fresh trace: start from 0 (runFrom(0) resets the global
  // store — leftover events from a previous view can never leak in, and
  // playback starts even if that view ended paused). Later runs resume
  // from whatever the store has applied; pausing stops the pending timer.
  const mounted = useRef(false);
  useEffect(() => {
    stateRef.current = { playing, speed };
    const from = mounted.current ? useTraceStore.getState().events.length : 0;
    mounted.current = true;
    if (playing) runFrom(from);
    return stopTimer;
  }, [playing, speed, runFrom]);

  return {
    playing,
    speed,
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    reset: () => {
      runFrom(0);
    },
    setSpeed: (s) => setSpeed(s),
  };
}

// ——— live: fetch-streamed SSE ————————————————————————

export interface LiveTraceOptions {
  prompt: string;
  maxTokens?: number;
  traceMode?: "BASIC" | "STANDARD" | "RESEARCH";
  model?: string;
  onTraceId?: (id: string) => void;
}

export function useLiveTrace() {
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const begin = useTraceStore((s) => s.begin);
  const accept = useTraceStore((s) => s.accept);
  const complete = useTraceStore((s) => s.complete);
  const fail = useTraceStore((s) => s.fail);

  const start = useCallback(
    async (opts: LiveTraceOptions) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setRunning(true);

      try {
        const res = await fetch(`${ENGINE_URL}/trace`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            prompt: opts.prompt,
            maxTokens: opts.maxTokens ?? 60,
            traceMode: opts.traceMode ?? "STANDARD",
            model: opts.model,
          }),
        });
        if (!res.ok || !res.body) {
          throw new Error(`engine responded ${res.status}`);
        }

        const parser = new SseParser();
        const reader = res.body.getReader();

        const dispatch = (event: string, data: string) => {
          if (event === "trace") {
            const envelope = JSON.parse(data) as Trace;
            opts.onTraceId?.(envelope.id);
            begin("live", envelope);
          } else if (event === "trace_event") {
            const parsed = traceEventSchema.safeParse(JSON.parse(data));
            if (parsed.success) accept(parsed.data as TraceEvent);
          } else if (event === "done") {
            complete();
          } else if (event === "error") {
            const err = JSON.parse(data) as { message?: string };
            fail(err.message ?? "generation failed");
          }
        };

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          for (const frame of parser.push(value)) {
            dispatch(frame.event, frame.data);
          }
        }
        for (const frame of parser.flush()) {
          dispatch(frame.event, frame.data);
        }
        complete();
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          fail((err as Error).message);
        }
      } finally {
        setRunning(false);
      }
    },
    [begin, accept, complete, fail],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setRunning(false);
  }, []);

  return { running, start, stop };
}
