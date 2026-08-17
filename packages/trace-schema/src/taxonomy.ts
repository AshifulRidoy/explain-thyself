/**
 * Signal taxonomy (spec §38–39): what each number in the interface IS.
 * Drives Inspector field badges and the /methodology page.
 *
 * Every signal shown in the UI must have an entry here — the vitest suite
 * enforces that Inspector-rendered field keys are covered.
 */
import type { EpistemicLevel } from "./events.js";

export interface SignalDefinition {
  level: EpistemicLevel;
  definition: string;
}

export const SIGNAL_TAXONOMY: Record<string, SignalDefinition> = {
  "input.tokens": {
    level: "MEASURED",
    definition: "Tokenizer output for the prompt; positions are absolute.",
  },
  "token.probability": {
    level: "MEASURED",
    definition: "Softmax of the model's logits at the sampled position.",
  },
  "token.rank": {
    level: "MEASURED",
    definition: "Rank of the emitted token in the full output distribution (0 = argmax).",
  },
  "token.entropyBits": {
    level: "DERIVED",
    definition:
      "Shannon entropy (log2) of the full next-token distribution. Low = the model's next move was nearly determined; high = genuinely contested.",
  },
  "token.latencyMs": {
    level: "MEASURED",
    definition: "Wall-clock forward pass for this step.",
  },
  "token.topK": {
    level: "MEASURED",
    definition: "Top-8 candidates of the output distribution with their probabilities.",
  },
  "layer.l2Norm": {
    level: "DERIVED",
    definition:
      "L2 norm of the residual stream at the final position of a layer. The underlying vector is measured; this scalar is its reduction.",
  },
  "layer.normRatio": {
    level: "DERIVED",
    definition:
      "l2Norm divided by the running mean of that layer's l2Norm over steps so far (first step = 1.0). >1 means the layer is more active than its recent average.",
  },
  "attention.weight": {
    level: "DERIVED",
    definition: "Mean over heads of attention from the final position to each prior position.",
  },
  "attention.headEntropyBits": {
    level: "DERIVED",
    definition: "Per-head entropy of a row of the attention pattern.",
  },
  "concept.score": {
    level: "INTERPRETED",
    definition:
      "Semantic-cluster association produced by the interpretation layer. Not a property of any single neuron.",
  },
  "decision.reason": {
    level: "DERIVED",
    definition: "Why generation ended or how the token was chosen (greedy/sampled/stop).",
  },
  "output.durationMs": {
    level: "MEASURED",
    definition: "Total wall-clock generation time.",
  },
};

export function taxonomyFor(key: string): SignalDefinition | undefined {
  return SIGNAL_TAXONOMY[key];
}
