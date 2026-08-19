"use client";

import type { TraceMode } from "@ets/trace-schema";
import type { ConceptActivity } from "@/lib/trace/store";

/**
 * "What concepts were most active?" — the Phase 5 INTERPRETED layer
 * (spec §20). Each row is a dictionary concept ranked by the total
 * probability mass the model's next-token distributions placed on its
 * words; the strip shows WHERE it was active. Italic serif labels mark
 * the whole ranking as interpretation — the mass is measured, the
 * label is authored.
 */
export function ConceptPanel({
  activities,
  tokenCount,
  traceMode,
}: {
  activities: ConceptActivity[];
  /** emitted tokens so far — normalizes the strip's x axis while streaming */
  tokenCount: number;
  traceMode?: TraceMode;
}) {
  const collected = traceMode !== "BASIC";
  const empty = activities.length === 0;

  return (
    <div data-testid="concept-panel">
      <div className="machine-label flex items-baseline justify-between">
        <span>What concepts were most active?</span>
        <span className="italic text-muted">interpreted</span>
      </div>

      {empty && (
        <p className="machine-label mt-3 text-muted" data-testid="concept-empty">
          {collected
            ? "Awaiting interpretation — no concept has crossed the mass threshold yet."
            : "Not collected — BASIC mode. Run STANDARD or RESEARCH to collect concepts."}
        </p>
      )}

      {!empty && (
        <ul className="mt-3 divide-y divide-line" data-testid="concept-rows">
          {activities.map((a) => (
            <li key={a.conceptId} className="py-2.5 first:pt-0" data-testid="concept-row">
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-serif text-sm italic leading-snug text-muted">
                  {a.label}
                </span>
                <span className="machine-label shrink-0 tabular-nums">
                  peak {a.peak.score.toFixed(2)} ·{" "}
                  {a.events.length === 1 ? "1 step" : `${a.events.length} steps`}
                </span>
              </div>
              <ConceptStrip activity={a} tokenCount={tokenCount} />
              {a.peak.evidence?.length ? (
                <div className="machine-label mt-1 truncate text-muted">
                  {a.peak.evidence.map((e) => e.text).join(" · ")}
                </div>
              ) : null}
            </li>
          ))}
        </ul>
      )}

      {!empty && (
        <p className="machine-label mt-3 text-muted">
          score = probability mass on the concept&rsquo;s words · the label is an
          interpretation
        </p>
      )}
    </div>
  );
}

/** Where in the trace the concept carried mass — one tick per active step. */
function ConceptStrip({
  activity,
  tokenCount,
}: {
  activity: ConceptActivity;
  tokenCount: number;
}) {
  const w = 220;
  const h = 18;
  const maxScore = Math.max(...activity.events.map((e) => e.score), 0.01);
  const span = Math.max(tokenCount - 1, 1);
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="mt-1.5 h-[18px] w-full border-b border-line"
      preserveAspectRatio="none"
      role="img"
      aria-label={`${activity.label} activity across the trace`}
    >
      {activity.events.map((e) => {
        const position = e.positions?.[e.positions.length - 1] ?? 0;
        const x = (position / span) * (w - 2) + 1;
        const height = Math.max((e.score / maxScore) * (h - 3), 1.5);
        return (
          <line
            key={e.id}
            x1={x}
            x2={x}
            y1={h - height}
            y2={h}
            stroke="var(--ink)"
            strokeOpacity={0.7}
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />
        );
      })}
    </svg>
  );
}
