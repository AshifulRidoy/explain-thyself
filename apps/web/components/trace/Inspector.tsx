import type {
  AttentionEvent,
  ConceptEvent,
  LayerActivityEvent,
  TraceEvent,
  TraceMode,
} from "@ets/trace-schema";
import { AttentionPanel } from "@/components/data-viz/AttentionPanel";
import { LayerActivityPanel } from "@/components/data-viz/LayerActivityPanel";
import { ProbabilityDistribution } from "@/components/data-viz/ProbabilityDistribution";
import {
  UNCERTAINTY_LABELS,
  uncertaintyTaxonomyKey,
} from "@/lib/trace/uncertainty";

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

function LevelBadge({ level }: { level: string | null }) {
  // null level = the quantity was considered and deliberately not measured
  return (
    <span
      className={`machine-label ${level ? (LEVEL_CLASS[level] ?? "text-muted") : "italic text-muted"}`}
    >
      {level ?? "not measured"}
    </span>
  );
}

export function Inspector({
  event,
  layerActivity,
  attention,
  concepts,
  traceMode,
}: {
  event: TraceEvent | null;
  /** LAYER_ACTIVITY paired to a selected TOKEN event, if present */
  layerActivity: LayerActivityEvent | null;
  /** ATTENTION events (all layers) paired to a selected TOKEN, if collected */
  attention?: AttentionEvent[] | null;
  /** CONCEPT events measured at the selected TOKEN's position, if any */
  concepts?: ConceptEvent[] | null;
  /** envelope mode — lets the panel say "RESEARCH only", not just "awaiting" */
  traceMode?: TraceMode;
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
          <AttentionPanel attention={attention ?? null} traceMode={traceMode} />
          {concepts && concepts.length > 0 && <ConceptsAtStep concepts={concepts} />}
        </>
      )}

      {event.type === "INPUT" && (
        <section className="divide-y divide-line border-t border-line">
          <Field taxonomyKey="input.tokens" label="Tokens" value={String(event.tokenCount)} />
        </section>
      )}

      {event.type === "CONCEPT" && (
        <>
          <section className="divide-y divide-line border-t border-line">
            <div className="flex items-baseline justify-between gap-4 py-1.5">
              <span className="machine-label shrink-0">Label</span>
              <span className="font-serif text-sm italic text-muted">
                {event.label}
              </span>
            </div>
            <Field
              taxonomyKey="concept.score"
              label="Mass"
              value={event.score.toFixed(4)}
              hint={
                event.positions?.length
                  ? `at position ${event.positions[event.positions.length - 1]}`
                  : undefined
              }
            />
          </section>
          {event.evidence?.length ? (
            <section className="border-t border-line">
              <div className="machine-label pt-2">
                Evidence — tokens carrying the mass
              </div>
              <ul className="mt-1">
                {event.evidence.map((evidence) => (
                  <li
                    key={`${evidence.tokenId}-${evidence.text}`}
                    className="flex items-baseline justify-between gap-4 border-b border-line py-1 last:border-b-0"
                  >
                    <span className="font-serif text-sm italic text-muted">
                      {evidence.text}
                    </span>
                    <span className="font-mono text-sm tabular-nums">
                      {evidence.probability.toFixed(4)}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </>
      )}

      {event.type === "OUTPUT" && (
        <section className="divide-y divide-line border-t border-line">
          <Field taxonomyKey="output.durationMs" label="Duration" value={`${event.durationMs} ms`} />
          <Field taxonomyKey="output.finishReason" label="Finish" value={event.finishReason} />
        </section>
      )}

      {event.type === "UNCERTAINTY" && (
        <>
          <section className="divide-y divide-line border-t border-line">
            <Field
              taxonomyKey={`uncertainty.${uncertaintyTaxonomyKey(event.kind)}`}
              label="Quantity"
              value={UNCERTAINTY_LABELS[event.kind]}
            />
            <Field
              taxonomyKey={`uncertainty.${uncertaintyTaxonomyKey(event.kind)}`}
              label="Value"
              value={event.value === null ? "—" : event.value.toFixed(4)}
              hint={
                event.window
                  ? `steps ${event.window.fromStep}–${event.window.toStep}`
                  : undefined
              }
            />
          </section>
          <section className="border-t border-line pt-2">
            <div className="machine-label">Basis</div>
            <p className="mt-1 font-serif text-sm italic leading-snug text-muted">
              {event.basis}
            </p>
          </section>
          {event.variants?.length ? (
            <section className="border-t border-line pt-2">
              <div className="machine-label">
                Variants — perturbed prompts actually rerun
              </div>
              <ul className="mt-1">
                {event.variants.map((v) => (
                  <li
                    key={v.perturbation}
                    className="border-b border-line py-1 last:border-b-0"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="machine-label">
                        {v.perturbation.replaceAll("_", " ")}
                      </span>
                      <span className="font-mono text-sm tabular-nums">
                        {v.agreedTokens}/{v.totalTokens}
                      </span>
                    </div>
                    <div className="machine-label truncate text-muted" title={v.text}>
                      &ldquo;{v.text}&rdquo;
                      {v.divergedPositions.length > 0 &&
                        ` · diverged at ${v.divergedPositions.join(", ")}`}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </>
      )}

      {event.type === "LAYER_ACTIVITY" && <LayerActivityPanel activity={event} />}

      {event.type === "ATTENTION" && (
        <AttentionPanel attention={[event]} traceMode="RESEARCH" />
      )}
    </div>
  );
}

/** The concepts measured at this step — mass + interpreted label pairs. */
function ConceptsAtStep({ concepts }: { concepts: ConceptEvent[] }) {
  return (
    <section className="border-t border-line pt-2" data-testid="token-concepts">
      <div className="machine-label">
        Concepts at this step <span className="italic text-muted">interpreted</span>
      </div>
      <ul className="mt-1">
        {concepts.map((c) => (
          <li
            key={c.id}
            className="flex items-baseline justify-between gap-4 border-b border-line py-1 last:border-b-0"
          >
            <span className="font-serif text-sm italic text-muted">{c.label}</span>
            <span className="font-mono text-sm tabular-nums">{c.score.toFixed(4)}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
