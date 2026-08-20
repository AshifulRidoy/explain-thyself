"use client";

/**
 * Counterfactual client (spec §23): POST /trace/:id/counterfactual streams
 * one comparison per edited prompt as SSE — the same parser the live trace
 * uses — so each variable lands in the panel when its rerun finishes.
 * Results persist engine-side; GET restores them for a replayed trace.
 */
import {
  counterfactualRequestSchema,
  counterfactualResultSchema,
  type CounterfactualRequest,
  type CounterfactualResult,
} from "@ets/trace-schema";
import { SseParser } from "./sse";

const ENGINE_URL =
  process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";

function parseResult(raw: unknown): CounterfactualResult {
  const parsed = counterfactualResultSchema.safeParse(raw);
  if (!parsed.success) {
    throw new Error(`counterfactual contract drift: ${parsed.error.message}`);
  }
  return parsed.data;
}

/** A trace's stored counterfactuals (latest investigation state). */
export async function fetchCounterfactuals(
  traceId: string,
): Promise<CounterfactualResult[]> {
  const res = await fetch(`${ENGINE_URL}/trace/${traceId}/counterfactuals`);
  if (!res.ok) throw new Error(`engine responded ${res.status}`);
  const data = (await res.json()) as { results: unknown[] };
  return data.results.map(parseResult);
}

/** Run counterfactuals; each result is handed to onResult as it streams. */
export async function runCounterfactuals(
  traceId: string,
  request: CounterfactualRequest,
  onResult: (result: CounterfactualResult) => void,
): Promise<CounterfactualResult[]> {
  // shared contract validates before it leaves the browser
  const body = counterfactualRequestSchema.parse(request);
  const res = await fetch(`${ENGINE_URL}/trace/${traceId}/counterfactual`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    const detail = res.status === 400 ? " — the engine rejected the request" : "";
    throw new Error(`engine responded ${res.status}${detail}`);
  }

  const parser = new SseParser();
  const reader = res.body.getReader();
  const collected: CounterfactualResult[] = [];
  const dispatch = (event: string, data: string) => {
    if (event === "counterfactual") {
      const result = parseResult(JSON.parse(data));
      collected.push(result);
      onResult(result);
    }
    // "done" ends the stream — anything else is ignored (heartbeats)
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const frame of parser.push(value)) dispatch(frame.event, frame.data);
  }
  for (const frame of parser.flush()) dispatch(frame.event, frame.data);
  return collected;
}

/**
 * Rank for the VARIABLE/IMPACT table (spec §23): newest run per edited
 * prompt wins (re-running an edit supersedes), then impact descending —
 * what would change the answer MOST sits on top.
 */
export function rankByImpact(results: CounterfactualResult[]): CounterfactualResult[] {
  const latest = new Map(results.map((r) => [r.promptText, r]));
  return [...latest.values()].sort((a, b) => b.impact - a.impact);
}
