/**
 * Signal taxonomy (spec §38–39): what each number in the interface IS.
 * Drives Inspector field badges and the /methodology page.
 *
 * Every signal shown in the UI must have an entry here — the vitest suite
 * enforces that Inspector-rendered field keys are covered.
 */
import type { EpistemicLevel } from "./events";

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
      "Probability mass the model's full next-token distribution places on a human-authored concept's dictionary words at one step. The mass is measured exactly; the label is an interpretation — not a property of any neuron, and not the model's thoughts.",
  },
  "uncertainty.modelUncertainty": {
    level: "DERIVED",
    definition:
      "Mean normalized entropy H/log2(V) over the emitted tokens. One number from one distribution — it cannot separate epistemic from aleatoric uncertainty; ensembles or samples would be needed for that.",
  },
  "uncertainty.answerStability": {
    level: "MEASURED",
    definition:
      "Fraction of greedy tokens that survived controlled surface perturbations of the prompt (strip final punctuation, lowercase first character), rerun to the same length. Low = the answer flips under input changes that carry no new meaning.",
  },
  "uncertainty.inputAmbiguity": {
    level: "INTERPRETED",
    definition:
      "Not measured. Estimating input ambiguity requires an auxiliary model or alternate-interpretation generation (spec §22); this instrument runs one small model and ships the null rather than a fake score.",
  },
  "uncertainty.evidenceQuality": {
    level: "MEASURED",
    definition:
      "Not measured. Evidence quality scores retrieval sources; this instrument has no retrieval and nothing to score — the null ships instead of an invented number.",
  },
  "counterfactual.variable": {
    level: "INTERPRETED",
    definition:
      "Human-authored label for the word substitution a counterfactual reruns (e.g. 'experience' for beginner→veteran). The label names the edit; the impact under it is measured. Not causal attribution — activation patching is the causal version, and it is future work.",
  },
  "counterfactual.impact": {
    level: "DERIVED",
    definition:
      "1 − token agreement between the original answer and the greedy rerun under one prompt edit, compared over the original's token count. 0 = the answer survived the edit unchanged; high = the edit flipped the answer early.",
  },
  "counterfactual.entropyDelta": {
    level: "DERIVED",
    definition:
      "Signed shift in mean per-token entropy between the original answer and the counterfactual rerun — did the edit make the model more or less uncertain while answering?",
  },
  "search.embedding": {
    level: "MEASURED",
    definition:
      "The model's own representation of a prompt: the mean of the final-layer residual stream over the prompt's tokens, L2-normalized (768 floats for GPT-2 small). Read inside the backend; the vector never leaves as a tensor. NULL rows are unsearchable, never approximated.",
  },
  "search.similarity": {
    level: "DERIVED",
    definition:
      "Cosine similarity between two measured prompt embeddings, computed in Postgres via pgvector. The RANK says how closely the model represents two prompts in its final layer — it is not semantic meaning, and absolute values are compressed by GPT-2's anisotropic hidden space.",
  },
  "comparison.agreement": {
    level: "DERIVED",
    definition:
      "Fraction of matching token ids over the first min(lenA, lenB) positions when the same prompt runs greedy through two models that share a tokenizer. Surface token overlap — not internal similarity; the models' internals are compared only through their own traces. Rejected outright when the tokenizers differ: agreement across incomparable ids would be a number about nothing.",
  },
  "comparison.entropyDelta": {
    level: "DERIVED",
    definition:
      "Signed shift in mean per-token entropy between the two models' answers to the same prompt — computed from the entropyBits each trace already shipped, so it is recomputable from the trace JSON. Different vocab-spread baselines make this a within-pair contrast, not a model-quality score.",
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
