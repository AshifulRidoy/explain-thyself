/**
 * Drizzle schema — the single DDL author for the whole system (docs/architecture.md).
 * The Python engine writes to these tables via asyncpg raw SQL; it never
 * issues CREATE/ALTER itself, and verifies this shape at startup.
 *
 * `payload` jsonb stores the exact validated TraceEvent JSON, so replay is
 * byte-faithful and the TS/Python read paths cannot disagree on shape.
 */
import {
  bigserial,
  index,
  integer,
  jsonb,
  pgTable,
  real,
  serial,
  text,
  timestamp,
} from "drizzle-orm/pg-core";

export const traces = pgTable("traces", {
  /** "tr_…" issued by the engine. */
  id: text("id").primaryKey(),
  /** DB serial → rendered as "TRACE 0042". */
  displayId: serial("display_id").notNull().unique(),
  /** Anonymous cookie id; no accounts in the MVP. */
  sessionId: text("session_id"),
  modelName: text("model_name").notNull(),
  modelRevision: text("model_revision").notNull().default(""),
  device: text("device").notNull().default(""),
  traceMode: text("trace_mode").notNull().default("STANDARD"),
  input: text("input").notNull(),
  outputText: text("output_text"),
  status: text("status").notNull().default("streaming"),
  maxTokens: integer("max_tokens").notNull().default(60),
  temperature: real("temperature").notNull().default(0),
  tokenCount: integer("token_count"),
  durationMs: integer("duration_ms"),
  /**
   * The stream-start envelope JSON (events removed) exactly as it was sent.
   * Replay reads this so model dims / sampling / revision survive the round
   * trip byte-faithfully. Nullable: pre-envelope rows derive what they can.
   */
  envelope: jsonb("envelope"),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

export const traceEvents = pgTable(
  "trace_events",
  {
    id: bigserial("id", { mode: "number" }).primaryKey(),
    traceId: text("trace_id")
      .notNull()
      .references(() => traces.id, { onDelete: "cascade" }),
    seq: integer("seq").notNull(),
    type: text("type").notNull(),
    level: text("level").notNull(),
    position: integer("position"),
    layer: integer("layer"),
    payload: jsonb("payload").notNull(),
    t: integer("t").notNull().default(0),
  },
  (table) => [index("trace_events_trace_seq_idx").on(table.traceId, table.seq)],
);

export const concepts = pgTable("concepts", {
  id: text("id").primaryKey(),
  traceId: text("trace_id")
    .notNull()
    .references(() => traces.id, { onDelete: "cascade" }),
  label: text("label").notNull(),
  score: real("score").notNull(),
  kind: text("kind").notNull().default("INTERPRETED"),
  payload: jsonb("payload"),
});

export type TraceRow = typeof traces.$inferSelect;
export type TraceEventRow = typeof traceEvents.$inferSelect;
