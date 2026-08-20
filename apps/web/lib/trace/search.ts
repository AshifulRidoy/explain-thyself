"use client";

/**
 * Trace search client (spec §28): the /traces page asks the engine to
 * embed the query with the SAME model that recorded the traces, then
 * ranks stored embeddings by cosine. The response validates against the
 * shared contract before it touches the UI.
 */
import { searchResponseSchema, type SearchResponse } from "@ets/trace-schema";

const ENGINE_URL =
  process.env.NEXT_PUBLIC_ENGINE_URL ?? "http://localhost:8000";

export async function searchTraces(q: string, limit = 10): Promise<SearchResponse> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  const res = await fetch(`${ENGINE_URL}/search?${params.toString()}`);
  if (!res.ok) throw new Error(`engine responded ${res.status}`);
  const parsed = searchResponseSchema.safeParse(await res.json());
  if (!parsed.success) {
    throw new Error(`search contract drift: ${parsed.error.message}`);
  }
  return parsed.data;
}
