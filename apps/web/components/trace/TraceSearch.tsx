"use client";

/**
 * Trace search (spec §28) for the /traces page: ask the engine which
 * recorded prompts the model represents like this one. Ranked by measured
 * cosine; the basis under the results is the contract's own string, so
 * the UI never invents a caption for the number.
 */

import { useState } from "react";
import Link from "next/link";
import { searchTraces } from "@/lib/trace/search";
import type { SearchResponse } from "@ets/trace-schema";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "results"; data: SearchResponse };

export function TraceSearch() {
  const [q, setQ] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });

  async function run(e: React.FormEvent) {
    e.preventDefault();
    const query = q.trim();
    if (!query) return;
    setState({ kind: "loading" });
    try {
      setState({ kind: "results", data: await searchTraces(query, 10) });
    } catch (err) {
      setState({
        kind: "error",
        message:
          err instanceof Error && err.message.includes("engine responded")
            ? "Trace search needs the engine — is it running on :8000?"
            : "Search failed.",
      });
    }
  }

  return (
    <div data-testid="trace-search" className="mb-10">
      <form onSubmit={run} className="flex items-baseline gap-3">
        <label htmlFor="trace-search-input" className="machine-label text-muted">
          Find traces
        </label>
        <input
          id="trace-search-input"
          data-testid="search-input"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="what prompt do you care about?"
          maxLength={2000}
          className="min-w-0 flex-1 border-b border-line bg-transparent pb-1 font-serif text-lg italic text-ink placeholder:text-muted/60 focus:border-signal focus:outline-none"
        />
        <button
          type="submit"
          className="machine-label text-muted underline decoration-line underline-offset-4 transition-colors hover:text-signal"
        >
          Search →
        </button>
      </form>

      {state.kind === "loading" && (
        <p className="machine-label mt-4 text-muted">Embedding the query…</p>
      )}

      {state.kind === "error" && (
        <p data-testid="search-error" className="machine-label mt-4 text-signal">
          {state.message}
        </p>
      )}

      {state.kind === "results" && (
        <div data-testid="search-results" className="mt-6">
          {state.data.results.length === 0 ? (
            <p data-testid="search-empty" className="border-y border-line py-8">
              <span className="font-serif text-lg italic text-muted">
                No recorded prompt the model represents like this one.
              </span>
              <span className="machine-label mt-2 block text-muted">
                {state.data.searchable} trace
                {state.data.searchable === 1 ? "" : "s"} searchable
              </span>
            </p>
          ) : (
            <ol className="border-t border-line">
              {state.data.results.map((hit) => (
                <li key={hit.traceId} data-testid="search-hit" className="border-b border-line">
                  <Link
                    href={`/trace/${hit.traceId}`}
                    className="grid grid-cols-[auto_1fr_6rem] items-baseline gap-x-4 px-2 py-3 transition-colors hover:bg-panel sm:grid-cols-[5rem_1fr_7rem_6rem]"
                  >
                    <span className="font-mono text-xs tabular-nums text-muted">
                      {String(hit.displayId).padStart(4, "0")}
                    </span>
                    <span className="truncate font-serif text-lg italic">
                      {hit.input}
                    </span>
                    <span className="hidden font-mono text-xs uppercase text-muted sm:block">
                      {hit.modelName} · {hit.traceMode.toLowerCase()}
                    </span>
                    <span className="text-right font-mono text-xs tabular-nums text-muted">
                      {hit.similarity >= 0 ? "+" : ""}
                      {hit.similarity.toFixed(4)}
                    </span>
                  </Link>
                </li>
              ))}
            </ol>
          )}
          <p data-testid="search-basis" className="machine-label mt-4 text-muted">
            {state.data.basis}
          </p>
        </div>
      )}
    </div>
  );
}
