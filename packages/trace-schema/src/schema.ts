/**
 * Zod schemas — the runtime validation layer of the Trace contract.
 * Strict everywhere: unknown fields fail loudly rather than drifting silently
 * between the TypeScript and Python sides of the contract.
 */
import { z } from "zod";
import type { Trace } from "./events";

export const epistemicLevelSchema = z.enum([
  "MEASURED",
  "DERIVED",
  "INTERPRETED",
]);

export const traceEventTypeSchema = z.enum([
  "INPUT",
  "TOKEN",
  "LAYER_ACTIVITY",
  "ATTENTION",
  "CONCEPT",
  "EVIDENCE",
  "HYPOTHESIS",
  "DECISION",
  "UNCERTAINTY",
  "OUTPUT",
]);

export const traceModeSchema = z.enum(["BASIC", "STANDARD", "RESEARCH"]);
export const traceStatusSchema = z.enum(["streaming", "complete", "error"]);

const probability = z.number().min(0).max(1);

export const topTokenSchema = z
  .object({
    tokenId: z.number().int().nonnegative(),
    text: z.string(),
    rawText: z.string(),
    leadingSpace: z.boolean(),
    probability,
    rank: z.number().int().nonnegative(),
  })
  .strict();

const traceEventBase = z
  .object({
    id: z.string().regex(/^evt_\d{4}$/),
    seq: z.number().int().nonnegative(),
    type: traceEventTypeSchema,
    t: z.number().int().nonnegative(),
  })
  .strict();

const inputEventSchema = traceEventBase.extend({
  type: z.literal("INPUT"),
  level: z.literal("MEASURED"),
  text: z.string(),
  tokenCount: z.number().int().nonnegative(),
  tokens: z
    .array(
      z
        .object({
          position: z.number().int().nonnegative(),
          tokenId: z.number().int().nonnegative(),
          text: z.string(),
        })
        .strict(),
    )
    .max(512),
});

const tokenEventSchema = traceEventBase.extend({
  type: z.literal("TOKEN"),
  level: z.literal("MEASURED"),
  position: z.number().int().nonnegative(),
  step: z.number().int().nonnegative(),
  tokenId: z.number().int().nonnegative(),
  text: z.string(),
  rawText: z.string(),
  leadingSpace: z.boolean(),
  probability,
  rank: z.number().int().nonnegative(),
  entropyBits: z.number().min(0).max(20),
  topK: z.array(topTokenSchema).min(1).max(8),
  latencyMs: z.number().nonnegative(),
});

const layerActivityEventSchema = traceEventBase.extend({
  type: z.literal("LAYER_ACTIVITY"),
  level: z.literal("DERIVED"),
  position: z.number().int().nonnegative(),
  step: z.number().int().nonnegative(),
  layers: z
    .array(
      z
        .object({
          layer: z.number().int().nonnegative(),
          l2Norm: z.number().positive(),
          normRatio: z.number().positive(),
        })
        .strict(),
    )
    .min(1),
});

const attentionEventSchema = traceEventBase.extend({
  type: z.literal("ATTENTION"),
  level: z.literal("DERIVED"),
  position: z.number().int().nonnegative(),
  layer: z.number().int().nonnegative(),
  aggregated: z
    .array(
      z
        .object({
          position: z.number().int().nonnegative(),
          text: z.string(),
          weight: probability,
        })
        .strict(),
    )
    .optional(),
  headEntropyBits: z.array(z.number().min(0).max(20)).optional(),
});

const conceptEventSchema = traceEventBase.extend({
  type: z.literal("CONCEPT"),
  level: z.literal("INTERPRETED"),
  conceptId: z.string(),
  label: z.string(),
  score: probability,
  positions: z.array(z.number().int().nonnegative()).optional(),
});

const evidenceEventSchema = traceEventBase.extend({
  type: z.literal("EVIDENCE"),
  level: z.literal("MEASURED"),
  evidenceId: z.string(),
  label: z.string(),
  source: z.string(),
  relevance: probability,
});

const hypothesisEventSchema = traceEventBase.extend({
  type: z.literal("HYPOTHESIS"),
  level: z.literal("INTERPRETED"),
  hypothesisId: z.string(),
  text: z.string(),
  confidence: probability,
});

const decisionEventSchema = traceEventBase.extend({
  type: z.literal("DECISION"),
  level: z.literal("DERIVED"),
  decision: z.enum(["sampled", "greedy", "stop", "max_tokens", "aborted"]),
  detail: z.string().optional(),
});

const uncertaintyEventSchema = traceEventBase.extend({
  type: z.literal("UNCERTAINTY"),
  level: z.literal("DERIVED"),
  kind: z.enum([
    "MODEL_UNCERTAINTY",
    "EVIDENCE_QUALITY",
    "INPUT_AMBIGUITY",
    "ANSWER_STABILITY",
  ]),
  value: z.number(),
  window: z
    .object({ fromStep: z.number().int().nonnegative(), toStep: z.number().int().nonnegative() })
    .strict()
    .optional(),
});

const outputEventSchema = traceEventBase.extend({
  type: z.literal("OUTPUT"),
  level: z.literal("MEASURED"),
  text: z.string(),
  tokenCount: z.number().int().nonnegative(),
  durationMs: z.number().nonnegative(),
  finishReason: z.string(),
});

export const traceEventSchema = z.discriminatedUnion("type", [
  inputEventSchema,
  tokenEventSchema,
  layerActivityEventSchema,
  attentionEventSchema,
  conceptEventSchema,
  evidenceEventSchema,
  hypothesisEventSchema,
  decisionEventSchema,
  uncertaintyEventSchema,
  outputEventSchema,
]);

export const traceSchema = z
  .object({
    id: z.string().regex(/^tr_[0-9a-z]{2,12}$/),
    displayId: z.number().int().positive(),
    model: z
      .object({
        name: z.string(),
        revision: z.string(),
        device: z.string(),
        layerCount: z.number().int().positive(),
        paramCount: z.number().int().positive(),
      })
      .strict(),
    input: z.object({ text: z.string() }).strict(),
    traceMode: traceModeSchema,
    sampling: z
      .object({
        maxTokens: z.number().int().positive(),
        temperature: z.number().min(0),
        topK: z.number().int().positive().nullable(),
        seed: z.number().int().nullable(),
      })
      .strict(),
    status: traceStatusSchema,
    createdAt: z.string().datetime(),
    output: z
      .object({
        text: z.string(),
        tokenCount: z.number().int().nonnegative(),
        durationMs: z.number().nonnegative(),
        finishReason: z.string(),
      })
      .strict()
      .optional(),
    events: z.array(traceEventSchema),
  })
  .strict();

/** Validate unknown JSON as a Trace at load boundaries (fixtures, API, DB reads). */
export function validateTrace(json: unknown): Trace {
  return traceSchema.parse(json);
}
