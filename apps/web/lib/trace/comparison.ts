"use client";

/**
 * Cross-model comparison client (spec Phase 7, V2 cut): POST /trace/:id/compare
 * runs the anchor trace's prompt through another registered model as its own
 * recorded trace and streams the derived comparison back as SSE — progress
 * frames carry model B's token count while it answers, one `comparison` frame
 * carries the artifact. GET restores stored comparisons for a replayed trace.
 */
import {
  comparisonRequestSchema,
  comparisonResultSchema,
  type ComparisonRequest,
  type ComparisonResult,
} from "@ets/trace-schema";
import { SseParser } from "./sse";

const ENGINE_URL =
  process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";

/** What GET /models lists per registered model (routes/meta.py). */
export interface ModelInfo {
  key: string;
  hfId: string;
  layerCount: number;
  headCount: number;
  dModel: number;
  paramCount: number;
  dtype: string;
  tokenizer: string;
}

function parseResult(raw: unknown): ComparisonResult {
  const parsed = comparisonResultSchema.safeParse(raw);
  if (!parsed.success) {
    throw new Error(`comparison contract drift: ${parsed.error.message}`);
  }
  return parsed.data;
}

/** A trace's stored comparisons (latest investigation state). */
export async function fetchComparisons(
  traceId: string,
): Promise<ComparisonResult[]> {
  const res = await fetch(`${ENGINE_URL}/trace/${traceId}/comparisons`);
  if (!res.ok) throw new Error(`engine responded ${res.status}`);
  const data = (await res.json()) as { results: unknown[] };
  return data.results.map(parseResult);
}

/** The registry, for the model picker. */
export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${ENGINE_URL}/models`);
  if (!res.ok) throw new Error(`engine responded ${res.status}`);
  const data = (await res.json()) as { models: ModelInfo[] };
  return data.models;
}

export interface ComparisonProgress {
  model: string;
  tokenCount: number;
}

/** Run one comparison; progress and the artifact stream as they land. */
export async function runComparison(
  traceId: string,
  request: ComparisonRequest,
  onProgress: (progress: ComparisonProgress) => void,
  onResult: (result: ComparisonResult) => void,
): Promise<ComparisonResult> {
  // shared contract validates before it leaves the browser
  const body = comparisonRequestSchema.parse(request);
  const res = await fetch(`${ENGINE_URL}/trace/${traceId}/compare`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    let detail = "";
    if (res.status === 400 || res.status === 404) {
      // the engine's honest rejections carry their reason
      try {
        detail = ` — ${((await res.json()) as { detail?: string }).detail ?? ""}`;
      } catch {
        /* non-JSON body; the status alone is the message */
      }
    }
    throw new Error(`engine responded ${res.status}${detail}`);
  }

  const parser = new SseParser();
  const reader = res.body.getReader();
  let result: ComparisonResult | null = null;
  const dispatch = (event: string, data: string) => {
    if (event === "progress") {
      onProgress(JSON.parse(data) as ComparisonProgress);
    } else if (event === "comparison") {
      result = parseResult(JSON.parse(data));
      onResult(result);
    } else if (event === "error") {
      const payload = JSON.parse(data) as { code?: string; message?: string };
      throw new Error(payload.message ?? payload.code ?? "comparison failed");
    }
    // "done" ends the stream — anything else is ignored (heartbeats)
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const frame of parser.push(value)) dispatch(frame.event, frame.data);
    }
    for (const frame of parser.flush()) dispatch(frame.event, frame.data);
  } finally {
    // an error frame leaves the stream open — release it
    void reader.cancel().catch(() => undefined);
  }
  if (!result) throw new Error("comparison stream ended without an artifact");
  return result;
}
