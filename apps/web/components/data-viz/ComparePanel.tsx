"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { ComparisonResult } from "@ets/trace-schema";
import {
  fetchComparisons,
  fetchModels,
  runComparison,
  type ComparisonProgress,
} from "@/lib/trace/comparison";

/**
 * "How does another model answer this?" (spec Phase 7, V2 cut) — the
 * second microscope slide. The anchor trace's prompt runs through
 * another REGISTERED model as its own recorded trace; the panel shows
 * both answers side by side with the measured token agreement.
 *
 * Honesty rules: the picker only offers models that share the anchor's
 * tokenizer (anything else, the engine refuses — ids across different
 * vocabularies are not comparable); agreement is surface token overlap,
 * NEVER internal similarity (footnote says so); the answers render
 * whitespace-pre-wrap so a degenerate continuation is visible as what
 * it is; each side keeps its own token count next to the shared score.
 */
export function ComparePanel({
  traceId,
  modelName,
  pending = false,
}: {
  /** engine-backed and completed (live after done, or replay); null otherwise */
  traceId: string | null;
  /** the anchor trace's model — from the envelope, never guessed */
  modelName: string | null;
  /** true while a live trace is still streaming */
  pending?: boolean;
}) {
  const [results, setResults] = useState<ComparisonResult[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<string[] | null>(null);
  const [pick, setPick] = useState("");
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ComparisonProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  // restore previous comparisons + the registry's comparable models
  useEffect(() => {
    setResults([]);
    setSelectedId(null);
    setCandidates(null);
    setPick("");
    if (!traceId || !modelName) return;
    let cancelled = false;
    fetchComparisons(traceId)
      .then((stored) => {
        if (cancelled) return;
        setResults(stored);
        setSelectedId(stored.at(-1)?.id ?? null);
      })
      .catch(() => {
        /* engine unreachable — the button will surface the error */
      });
    fetchModels()
      .then((models) => {
        if (cancelled) return;
        // only models that share the anchor's tokenizer are offered;
        // the engine would reject the rest (and says why)
        const anchor = models.find((m) => m.key === modelName);
        const comparable = anchor
          ? models.filter(
              (m) => m.key !== modelName && m.tokenizer === anchor.tokenizer,
            )
          : [];
        setCandidates(comparable.map((m) => m.key));
        setPick(comparable[0]?.key ?? "");
      })
      .catch(() => {
        /* engine unreachable — the run button will surface the error */
      });
    return () => {
      cancelled = true;
    };
  }, [traceId, modelName]);

  const selected = useMemo(
    () => results.find((r) => r.id === selectedId) ?? null,
    [results, selectedId],
  );

  async function compare() {
    if (!traceId || !pick || running) return;
    setRunning(true);
    setProgress(null);
    setError(null);
    try {
      const result = await runComparison(
        traceId,
        { model: pick },
        setProgress,
        (r) => {
          setResults((prev) => [...prev, r]);
          setSelectedId(r.id);
        },
      );
      setSelectedId(result.id);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
      setProgress(null);
    }
  }

  return (
    <div data-testid="compare-panel">
      <div className="machine-label flex items-baseline justify-between">
        <span>How does another model answer this?</span>
        <span className="italic text-muted">comparison</span>
      </div>

      {traceId === null && (
        <p className="machine-label mt-3 text-muted" data-testid="compare-empty">
          {pending
            ? "Waiting for the model's answer — a comparison needs a finished answer."
            : "Needs the trace engine — a committed fixture cannot load another model. Run live or open a replay."}
        </p>
      )}

      {traceId !== null && (
        <>
          <div className="mt-3 flex items-center justify-between gap-3">
            <p className="machine-label text-muted">
              {running && progress
                ? `${progress.model} answering… ${progress.tokenCount} tokens`
                : running
                  ? "Loading the model…"
                  : results.length === 0
                    ? "Same prompt, second registered model, both answers recorded."
                    : "Each run records model B's answer as its own trace."}
            </p>
            {candidates !== null && candidates.length > 0 && (
              <div className="flex shrink-0 items-center gap-2">
                <select
                  value={pick}
                  onChange={(e) => setPick(e.target.value)}
                  disabled={running}
                  data-testid="compare-model-select"
                  aria-label="model to compare against"
                  className="machine-label border-b border-line bg-paper pb-0.5 text-ink focus:border-ink focus:outline-none"
                >
                  {candidates.map((key) => (
                    <option key={key} value={key}>
                      {key}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => void compare()}
                  disabled={running}
                  data-testid="compare-run"
                  className="machine-label shrink-0 border-b border-ink pb-0.5 text-ink transition-colors hover:text-signal disabled:border-line disabled:text-muted"
                >
                  {results.length === 0 ? "Compare →" : "↻ Again"}
                </button>
              </div>
            )}
            {candidates !== null && candidates.length === 0 && (
              <span className="machine-label shrink-0 text-muted">
                no comparable model registered
              </span>
            )}
          </div>

          {error && <p className="machine-label mt-2 text-signal">{error}</p>}

          {results.length > 1 && (
            <ul className="mt-3 divide-y divide-line" data-testid="compare-rows">
              {[...results].reverse().map((r) => (
                <li key={r.id}>
                  <button
                    onClick={() => setSelectedId(r.id === selectedId ? null : r.id)}
                    className="flex w-full items-baseline justify-between gap-3 py-2 text-left"
                    data-testid="compare-row"
                  >
                    <span className="machine-label">
                      {r.modelA} × {r.modelB}
                    </span>
                    <span className="machine-label text-muted tabular-nums">
                      {r.agreedTokens}/{r.comparedLength} agree ·{" "}
                      <span data-testid="compare-row-agreement">
                        {(r.agreement * 100).toFixed(1)}%
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}

          {selected && (
            <div className="mt-3 border-t border-line pt-3" data-testid="compare-result">
              {/* side by side; mobile stacks — never squeezed columns */}
              <div className="grid gap-3 sm:grid-cols-2">
                <div>
                  <div className="machine-label" data-testid="compare-output-a">
                    {selected.modelA} · {selected.tokenCountA} tokens
                  </div>
                  <p className="mt-1 whitespace-pre-wrap border border-line bg-panel p-2 font-mono text-xs leading-relaxed">
                    {selected.outputTextA}
                  </p>
                </div>
                <div>
                  <div className="machine-label" data-testid="compare-output-b">
                    {selected.modelB} · {selected.tokenCountB} tokens
                  </div>
                  <p className="mt-1 whitespace-pre-wrap border border-line bg-panel p-2 font-mono text-xs leading-relaxed">
                    {selected.outputTextB}
                  </p>
                </div>
              </div>

              <div className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <span className="font-mono text-sm tabular-nums" data-testid="compare-agreement">
                  {(selected.agreement * 100).toFixed(1)}%
                </span>
                <span className="machine-label text-muted tabular-nums">
                  {selected.agreedTokens}/{selected.comparedLength} compared positions agree
                </span>
                <span className="machine-label text-muted">
                  {selected.firstDivergence === null
                    ? "identical over the compared range"
                    : `first change at token ${selected.firstDivergence}`}
                </span>
                <span className="machine-label text-muted tabular-nums">
                  Δ entropy {selected.entropyDelta >= 0 ? "+" : ""}
                  {selected.entropyDelta.toFixed(2)} bits
                </span>
              </div>

              <p className="machine-label mt-2">
                <Link
                  href={`/trace/${selected.traceIdB}`}
                  className="text-ink underline decoration-line underline-offset-4 hover:text-signal"
                  data-testid="compare-replay-link"
                >
                  open {selected.modelB}&apos;s trace ↗
                </Link>
              </p>

              <p className="machine-label mt-2 leading-snug text-muted" data-testid="compare-basis">
                {selected.basis}
              </p>
            </div>
          )}

          {results.length > 0 && (
            <p className="machine-label mt-3 text-muted">
              shared token overlap under a common tokenizer · surface behavior, not internal similarity
            </p>
          )}
        </>
      )}
    </div>
  );
}
