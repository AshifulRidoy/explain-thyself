"use client";

import type { TraceMode, UncertaintyEvent } from "@ets/trace-schema";
import { UNCERTAINTY_LABELS } from "@/lib/trace/uncertainty";

/**
 * "What is the model uncertain about?" — the spec §22 layer: four SEPARATE
 * quantities, never one blended confidence number. Measured rows carry a
 * value and their level; the two this instrument cannot measure honestly
 * show the refusal in the row itself (value —, reason beneath), because
 * the trace ships them as nulls with a basis, not as UI copy.
 */
export function UncertaintyPanel({
  quantities,
  traceMode,
  complete,
}: {
  /** in spec display order (MODEL, EVIDENCE, INPUT, STABILITY) */
  quantities: UncertaintyEvent[];
  traceMode?: TraceMode;
  /** uncertainty is measured after the answer exists */
  complete?: boolean;
}) {
  const collected = traceMode === "RESEARCH";
  const empty = quantities.length === 0;

  return (
    <div data-testid="uncertainty-panel">
      <div className="machine-label flex items-baseline justify-between">
        <span>What is the model uncertain about?</span>
        <span className="italic text-muted">separated</span>
      </div>

      {empty && (
        <p className="machine-label mt-3 text-muted" data-testid="uncertainty-empty">
          {collected
            ? complete
              ? "No uncertainty events — the trace ended before analysis."
              : "Awaiting the answer — uncertainty is measured after generation completes."
            : "Not collected — RESEARCH mode. Answer stability re-runs the answer per prompt perturbation."}
        </p>
      )}

      {!empty && (
        <ul className="mt-3 divide-y divide-line" data-testid="uncertainty-rows">
          {quantities.map((q) => (
            <li key={q.id} className="py-2.5 first:pt-0" data-testid="uncertainty-row">
              <div className="flex items-baseline justify-between gap-3">
                <span className="machine-label shrink-0">
                  {UNCERTAINTY_LABELS[q.kind]}
                </span>
                <span className="flex items-baseline gap-2">
                  {q.value === null ? (
                    <span className="font-mono text-sm text-muted">—</span>
                  ) : (
                    <span className="font-mono text-sm tabular-nums">
                      {q.value.toFixed(2)}
                    </span>
                  )}
                  <span
                    className={`machine-label ${
                      q.level === "MEASURED"
                        ? "text-ink"
                        : q.level === null
                          ? "italic text-muted"
                          : "text-muted"
                    }`}
                  >
                    {q.level?.toLowerCase() ?? "not measured"}
                  </span>
                </span>
              </div>
              {/* the basis is the honesty: method for a value, reason for a null */}
              <p className="machine-label mt-1 leading-snug text-muted">{q.basis}</p>
              {q.variants?.length ? (
                <ul className="mt-1.5">
                  {q.variants.map((v) => (
                    <li
                      key={v.perturbation}
                      className="machine-label flex items-baseline justify-between gap-3 text-muted"
                      data-testid="uncertainty-variant"
                    >
                      <span className="truncate" title={v.text}>
                        {v.perturbation.replaceAll("_", " ")}
                      </span>
                      <span className="shrink-0 tabular-nums">
                        {v.agreedTokens}/{v.totalTokens} agree
                      </span>
                    </li>
                  ))}
                </ul>
              ) : null}
            </li>
          ))}
        </ul>
      )}

      {!empty && (
        <p className="machine-label mt-3 text-muted">
          four separate quantities · never one confidence number
        </p>
      )}
    </div>
  );
}
