/**
 * Replay read path: Next server components read Postgres via Drizzle
 * (Architecture.md — Drizzle owns DDL; reads split naturally between
 * runtimes). Mirrors services/trace-engine/app/storage/trace_reader.py
 * exactly: prefer the stored stream-start envelope, override final state
 * from columns, take finishReason from the OUTPUT event — never guess.
 *
 * Every trace is validated against the zod contract before it reaches a
 * page: the DB stores jsonb payloads precisely so this check can pass.
 */
import { desc, eq, sql } from "drizzle-orm";
import { traceSchema, type Trace } from "@ets/trace-schema";
import { db } from "@/lib/db";
import { traceEvents, traces } from "@/lib/db/schema";

export interface TraceSummary {
  id: string;
  displayId: number;
  input: string;
  status: string;
  traceMode: string;
  modelName: string;
  tokenCount: number | null;
  durationMs: number | null;
  createdAt: Date;
}

export type LoadResult =
  | { ok: true; trace: Trace }
  | { ok: false; reason: "not_found" | "invalid" | "db" };

export async function listRecentTraces(limit = 50): Promise<TraceSummary[]> {
  const rows = await db
    .select({
      id: traces.id,
      displayId: traces.displayId,
      input: traces.input,
      status: traces.status,
      traceMode: traces.traceMode,
      modelName: traces.modelName,
      tokenCount: traces.tokenCount,
      durationMs: traces.durationMs,
      createdAt: traces.createdAt,
    })
    .from(traces)
    .orderBy(desc(traces.createdAt), desc(traces.displayId))
    .limit(limit);
  return rows;
}

export interface SimilarTrace {
  traceId: string;
  displayId: number;
  input: string;
  similarity: number;
  modelName: string;
  traceMode: string;
  tokenCount: number | null;
}

/**
 * Spec §28 "find traces similar to this one", read through Drizzle so the
 * replay page stays engine-independent. Mirrors trace_reader.similar_traces
 * (cosine via pgvector `<=>`), including the model_name filter — each
 * backend has its own representation space, so a cosine across models
 * compares unrelated geometries. The source vector is passed as a
 * parameterized pgvector text literal, never interpolated raw.
 * `null` = unknown trace; an empty list = known trace with no embedding.
 */
export async function listSimilarTraces(
  id: string,
  limit = 5,
): Promise<SimilarTrace[] | null> {
  let source: { embedding: unknown; modelName: string } | undefined;
  try {
    [source] = await db
      .select({ embedding: traces.embedding, modelName: traces.modelName })
      .from(traces)
      .where(eq(traces.id, id))
      .limit(1);
  } catch {
    return null;
  }
  if (!source) return null;
  const literal = source.embedding as unknown; // string[] | string | null by driver
  const text =
    literal === null
      ? null
      : Array.isArray(literal)
        ? `[${literal.join(",")}]`
        : String(literal);
  if (text === null) return []; // recorded before search existed — honestly empty

  try {
    const result = await db.execute(sql`
      SELECT id, display_id, input, model_name, trace_mode, token_count,
             1 - (embedding <=> ${text}::vector) AS similarity
      FROM traces
      WHERE embedding IS NOT NULL AND model_name = ${source.modelName}
        AND id != ${id}
      ORDER BY embedding <=> ${text}::vector
      LIMIT ${limit}
    `);
    const rows = (result as unknown as { rows: Record<string, unknown>[] }).rows;
    return rows.map((r) => ({
      traceId: String(r.id),
      displayId: Number(r.display_id),
      input: String(r.input),
      similarity: Math.round(Number(r.similarity) * 10000) / 10000,
      modelName: String(r.model_name),
      traceMode: String(r.trace_mode),
      tokenCount: r.token_count === null ? null : Number(r.token_count),
    }));
  } catch {
    return null;
  }
}

export async function loadTraceById(id: string): Promise<LoadResult> {
  let row;
  let eventRows;
  try {
    [row] = await db.select().from(traces).where(eq(traces.id, id)).limit(1);
    if (!row) return { ok: false, reason: "not_found" };
    eventRows = await db
      .select({ payload: traceEvents.payload, seq: traceEvents.seq })
      .from(traceEvents)
      .where(eq(traceEvents.traceId, id))
      .orderBy(traceEvents.seq);
  } catch {
    return { ok: false, reason: "db" };
  }

  const events = eventRows.map((e) => e.payload);

  // stored envelope (exact) — fall back to flat columns for rows written
  // before the envelope column existed
  const base = (row.envelope as Record<string, unknown> | null) ?? {
    id: row.id,
    displayId: row.displayId,
    model: {
      name: row.modelName,
      revision: row.modelRevision,
      device: row.device,
      layerCount: 0,
      paramCount: 0,
    },
    input: { text: row.input },
    traceMode: row.traceMode,
    sampling: {
      maxTokens: row.maxTokens,
      temperature: row.temperature,
      topK: null,
      seed: null,
    },
    createdAt: row.createdAt.toISOString(),
  };

  const rebuilt = {
    ...base,
    status: row.status, // authoritative final state lives in columns
    ...(row.outputText !== null
      ? {
          output: {
            text: row.outputText,
            tokenCount: row.tokenCount ?? 0,
            durationMs: row.durationMs ?? 0,
            finishReason:
              (events.find(
                (e) => (e as { type?: string }).type === "OUTPUT",
              ) as { finishReason?: string } | undefined)?.finishReason ??
              "stop",
          },
        }
      : {}),
    events,
  } as Record<string, unknown>;

  const parsed = traceSchema.safeParse(rebuilt);
  if (!parsed.success) return { ok: false, reason: "invalid" };
  return { ok: true, trace: parsed.data };
}
