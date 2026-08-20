"use client";

import { useEffect, useMemo, useState } from "react";
import type { CounterfactualRequest, CounterfactualResult } from "@ets/trace-schema";
import {
  fetchCounterfactuals,
  rankByImpact,
  runCounterfactuals,
} from "@/lib/trace/counterfactuals";

/**
 * "What would change the answer?" (spec §23) — the investigation tool.
 * Each dictionary variable reruns the model on an edited prompt and is
 * compared token-by-token against the original answer; the table ranks
 * variables by measured impact. The free-form editor is the spec's
 * CounterfactualSlider spirit: manipulate the prompt, watch behavior move.
 *
 * Honesty rules: impact is word-substitution sensitivity, NEVER causal
 * attribution (footnote says so); the variable label is INTERPRETED while
 * the numbers under it are DERIVED from measured agreement.
 */
export function CounterfactualPanel({
  traceId,
  prompt,
  pending = false,
}: {
  /** engine-backed and completed (live after done, or replay); null otherwise */
  traceId: string | null;
  prompt: string;
  /** true while a live trace is still streaming — its answer doesn't exist yet */
  pending?: boolean;
}) {
  const [results, setResults] = useState<CounterfactualResult[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [edit, setEdit] = useState(prompt);

  // restore a previous investigation (results persist with the trace)
  useEffect(() => {
    setEdit(prompt);
    if (!traceId) return;
    let cancelled = false;
    fetchCounterfactuals(traceId)
      .then((stored) => {
        if (!cancelled) setResults(stored);
      })
      .catch(() => {
        /* engine unreachable — the button will surface the error */
      });
    return () => {
      cancelled = true;
    };
  }, [traceId, prompt]);

  const ranked = useMemo(() => rankByImpact(results), [results]);
  const selected = ranked.find((r) => r.id === selectedId) ?? null;

  async function investigate(request: CounterfactualRequest) {
    if (!traceId || running) return;
    setRunning(true);
    setError(null);
    try {
      await runCounterfactuals(traceId, request, (result) => {
        setResults((prev) => [...prev, result]);
        setSelectedId((current) => current ?? result.id);
      });
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }

  const editedPrompt = edit.trim();

  return (
    <div data-testid="counterfactual-panel">
      <div className="machine-label flex items-baseline justify-between">
        <span>What would change the answer?</span>
        <span className="italic text-muted">counterfactual</span>
      </div>

      {traceId === null && (
        <p className="machine-label mt-3 text-muted" data-testid="counterfactual-empty">
          {pending
            ? "Awaiting the answer — counterfactuals compare against a completed answer."
            : "Needs the trace engine — a committed fixture cannot re-run the model. Run live or open a replay."}
        </p>
      )}

      {traceId !== null && (
        <>
          <div className="mt-3 flex items-center justify-between gap-3">
            <p className="machine-label text-muted">
              {running
                ? "Investigating — each variable re-runs the model…"
                : ranked.length === 0
                  ? "Dictionary variables found in the prompt, rerun one word at a time."
                  : "Ranked by measured impact on the answer."}
            </p>
            <button
              onClick={() => void investigate({ scope: "all" })}
              disabled={running}
              data-testid="counterfactual-run"
              className="machine-label shrink-0 border-b border-ink pb-0.5 text-ink transition-colors hover:text-signal disabled:border-line disabled:text-muted"
            >
              {ranked.length === 0 ? "Investigate →" : "↻ Re-run"}
            </button>
          </div>

          {error && (
            <p className="machine-label mt-2 text-signal">{error}</p>
          )}

          {ranked.length > 0 && (
            <ul className="mt-3 divide-y divide-line" data-testid="counterfactual-rows">
              {ranked.map((r, i) => (
                <li key={r.id} className="py-2.5 first:pt-0" data-testid="counterfactual-row">
                  <button
                    onClick={() => setSelectedId(r.id === selectedId ? null : r.id)}
                    className="w-full text-left"
                    data-testid="counterfactual-row-button"
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="machine-label shrink-0">
                        {r.variable}
                        {r.originalWord && (
                          <span className="font-serif italic text-muted">
                            {" "}
                            · {r.originalWord} → {r.replacementWord}
                          </span>
                        )}
                      </span>
                      <span className="flex shrink-0 items-baseline gap-2">
                        <span className="machine-label text-muted tabular-nums">
                          {r.agreedTokens}/{r.tokenCount} agree
                        </span>
                        <span className="font-mono text-sm tabular-nums">
                          {r.impact.toFixed(2)}
                        </span>
                      </span>
                    </div>
                    {/* impact bar — Signal marks the top mover only */}
                    <div className="mt-1.5 h-1 w-full border-b border-line">
                      <div
                        className={`h-full ${i === 0 && r.impact > 0 ? "bg-signal" : "bg-ink"}`}
                        style={{ width: `${Math.round(r.impact * 100)}%` }}
                      />
                    </div>
                    <div className="machine-label mt-1 text-muted">
                      Δ entropy {r.entropyDelta >= 0 ? "+" : ""}
                      {r.entropyDelta.toFixed(2)} bits
                    </div>
                  </button>

                  {r.id === selectedId && (
                    <div className="mt-2 border-t border-line pt-2" data-testid="counterfactual-detail">
                      <div className="machine-label">Original</div>
                      <p className="font-serif text-sm italic text-muted">{prompt}</p>
                      <div className="machine-label mt-2">Counterfactual</div>
                      <p className="font-serif text-sm italic text-muted">{r.promptText}</p>
                      <p className="mt-2 font-serif text-sm leading-snug">{r.outputText}</p>
                      <p className="machine-label mt-2 text-muted">
                        {r.firstDivergence === null
                          ? "the answer survived this edit unchanged"
                          : `first change at token ${r.firstDivergence}`}
                      </p>
                      <p className="machine-label mt-1 leading-snug text-muted">{r.basis}</p>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}

          {/* the free-form edit — manipulate any variable, not just dictionary ones */}
          <div className="mt-3 border-t border-line pt-3">
            <label className="machine-label text-muted" htmlFor="counterfactual-edit">
              Edit the prompt and re-run
            </label>
            <textarea
              id="counterfactual-edit"
              data-testid="counterfactual-edit"
              value={edit}
              onChange={(e) => setEdit(e.target.value)}
              rows={2}
              className="mt-1 w-full resize-none border border-line bg-paper p-2 font-mono text-xs leading-relaxed focus:outline-none focus:border-ink"
            />
            <button
              onClick={() => void investigate({ scope: "prompt", prompt: editedPrompt })}
              disabled={running || editedPrompt === prompt.trim()}
              data-testid="counterfactual-edit-run"
              className="machine-label mt-1 border-b border-ink pb-0.5 text-ink transition-colors hover:text-signal disabled:border-line disabled:text-muted"
            >
              Re-run edited prompt
            </button>
          </div>

          {ranked.length > 0 && (
            <p className="machine-label mt-3 text-muted">
              word-substitution sensitivity, measured by greedy rerun · not causal attribution
            </p>
          )}
        </>
      )}
    </div>
  );
}
