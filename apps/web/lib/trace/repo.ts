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
import { desc, eq } from "drizzle-orm";
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
