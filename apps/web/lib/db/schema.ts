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
  vector,
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
  /**
   * The model's own final-layer prompt representation (spec §28 trace
   * search): mean of resid_post over prompt tokens, L2-normalized, 768
   * floats for GPT-2 small. NULL on rows recorded before this column or
   * when the embedding pass failed — such rows are simply unsearchable,
   * never silently approximated. Similarity is cosine, computed in
   * Postgres via pgvector (`<=>`); no ANN index yet (exact scan is
   * honest and fast at this scale).
   */
  embedding: vector("embedding", { dimensions: 768 }),
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

/**
 * One counterfactual comparison per row (spec §23): an edited prompt rerun
 * greedy, compared token-by-token against the original trace's answer.
 * The original trace stays immutable — these are attached artifacts, not
 * events. `payload` is the exact validated CounterfactualResult JSON.
 */
export const counterfactuals = pgTable(
  "counterfactuals",
  {
    /** "cf_…" issued by the engine. */
    id: text("id").primaryKey(),
    traceId: text("trace_id")
      .notNull()
      .references(() => traces.id, { onDelete: "cascade" }),
    /** INTERPRETED variable label ("experience") or "your edit". */
    variable: text("variable").notNull(),
    originalWord: text("original_word"),
    replacementWord: text("replacement_word"),
    /** the edited prompt that was rerun. */
    promptText: text("prompt_text").notNull(),
    /** 1 − token agreement; indexed column for ranking, payload is truth. */
    impact: real("impact").notNull(),
    payload: jsonb("payload").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (table) => [index("counterfactuals_trace_idx").on(table.traceId)],
);

/**
 * One cross-model comparison per row (spec Phase 7): the same prompt run
 * through two registered models that share a tokenizer. Both sides are
 * full persisted traces (traceIdA = anchor, traceIdB = fresh run); the
 * comparison itself is a separate artifact — the same immutability rule
 * as counterfactuals. `payload` is the exact validated ComparisonResult
 * JSON; agreement is denormalized for ranking.
 */
export const comparisons = pgTable(
  "comparisons",
  {
    /** "cmp_…" issued by the engine. */
    id: text("id").primaryKey(),
    /** the anchor trace the comparison hangs off. */
    traceIdA: text("trace_id_a")
      .notNull()
      .references(() => traces.id, { onDelete: "cascade" }),
    /** the freshly recorded run through model B. */
    traceIdB: text("trace_id_b")
      .notNull()
      .references(() => traces.id, { onDelete: "cascade" }),
    modelA: text("model_a").notNull(),
    modelB: text("model_b").notNull(),
    /** agreed/comparedLength; indexed column for ranking, payload is truth. */
    agreement: real("agreement").notNull(),
    payload: jsonb("payload").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
  },
  (table) => [index("comparisons_trace_a_idx").on(table.traceIdA)],
);

export type TraceRow = typeof traces.$inferSelect;
export type TraceEventRow = typeof traceEvents.$inferSelect;
export type CounterfactualRow = typeof counterfactuals.$inferSelect;
export type ComparisonRow = typeof comparisons.$inferSelect;
