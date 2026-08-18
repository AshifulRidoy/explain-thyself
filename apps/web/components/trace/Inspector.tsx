import type { LayerActivityEvent, TraceEvent } from "@ets/trace-schema";
import { LayerActivityPanel } from "@/components/data-viz/LayerActivityPanel";
import { ProbabilityDistribution } from "@/components/data-viz/ProbabilityDistribution";

/**
 * The Inspector: whatever is selected, shown with its epistemic badge from
 * SIGNAL_TAXONOMY. INTERPRETED rows render muted+italic — an interpretation
 * is not a measurement.
 */

const LEVEL_CLASS: Record<string, string> = {
  MEASURED: "text-ink",
  DERIVED: "text-muted",
  INTERPRETED: "text-muted italic",
};

function Field({
  taxonomyKey,
  label,
  value,
  hint,
}: {
  taxonomyKey: string;
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <span className="machine-label shrink-0">{label}</span>
      <span className="text-right">
        <span className="font-mono text-sm tabular-nums">{value}</span>
        {hint && <span className="block font-mono text-machine text-muted">{hint}</span>}
        <span className="sr-only">{taxonomyKey}</span>
      </span>
    </div>
  );
}

function LevelBadge({ level }: { level: string }) {
  return (
    <span className={`machine-label ${LEVEL_CLASS[level] ?? "text-muted"}`}>
      {level}
    </span>
  );
}

export function Inspector({
  event,
  layerActivity,
}: {
  event: TraceEvent | null;
  /** LAYER_ACTIVITY paired to a selected TOKEN event, if present */
  layerActivity: LayerActivityEvent | null;
}) {
  if (!event) {
    return (
      <p className="machine-label">Select a node to inspect its signals</p>
    );
  }

  return (
    <div className="space-y-8" data-testid="inspector">
      <header>
        <div className="flex items-baseline justify-between">
          <h3 className="font-mono text-sm uppercase tracking-widest">
            {event.type}
          </h3>
          <LevelBadge level={event.level} />
        </div>
        <div className="machine-label mt-1">
          {event.id} · t {event.t}ms
        </div>
      </header>

      {event.type === "TOKEN" && (
        <>
          <section className="divide-y divide-line border-t border-line">
            <Field taxonomyKey="token.text" label="Token" value={event.rawText} hint={`id ${event.tokenId}`} />
            <Field taxonomyKey="token.probability" label="Probability" value={event.probability.toFixed(4)} hint={`rank ${event.rank}`} />
            <Field taxonomyKey="token.entropyBits" label="Entropy" value={`${event.entropyBits.toFixed(2)} bits`} />
            <Field taxonomyKey="token.latencyMs" label="Latency" value={`${event.latencyMs.toFixed(1)} ms`} />
          </section>
          <ProbabilityDistribution topK={event.topK} sampledTokenId={event.tokenId} />
          <LayerActivityPanel activity={layerActivity} />
        </>
      )}

      {event.type === "INPUT" && (
        <section className="divide-y divide-line border-t border-line">
          <Field taxonomyKey="input.tokens" label="Tokens" value={String(event.tokenCount)} />
        </section>
      )}

      {event.type === "CONCEPT" && (
        <section className="divide-y divide-line border-t border-line">
          <Field taxonomyKey="concept.score" label="Score" value={event.score.toFixed(2)} />
        </section>
      )}

      {event.type === "OUTPUT" && (
        <section className="divide-y divide-line border-t border-line">
          <Field taxonomyKey="output.durationMs" label="Duration" value={`${event.durationMs} ms`} />
          <Field taxonomyKey="output.finishReason" label="Finish" value={event.finishReason} />
        </section>
      )}

      {event.type === "LAYER_ACTIVITY" && <LayerActivityPanel activity={event} />}
    </div>
  );
}
