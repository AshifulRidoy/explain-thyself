"use client";

import { useState } from "react";
import type { AttentionEvent, TraceMode } from "@ets/trace-schema";

/**
 * Head-mean attention FROM the selected position TO every prior position —
 * the hook-side reduction, never a pattern tensor (spec §19). One row per
 * context entry; position −1 is the prepended BOS token GPT-2 sinks
 * attention into — a real measurement, shown rather than hidden.
 *
 * DERIVED: 12-head mean of softmax rows (weights sum to 1) + per-head
 * entropy. Bars are scaled to the row max for shape; the number is the
 * measurement.
 */
export function AttentionPanel({
  attention,
  traceMode,
}: {
  /** all layers for one position, layer-ascending; null/[] while awaiting */
  attention: AttentionEvent[] | null;
  /** from the envelope — distinguishes "not yet" from "not collected" */
  traceMode?: TraceMode;
}) {
  const [layerSel, setLayerSel] = useState<number | null>(null);
  // events stream in L01…L12; fall back to the deepest layer arrived
  const chosen =
    attention?.find((a) => a.layer === layerSel) ?? attention?.at(-1) ?? null;
  const rows = chosen?.aggregated ?? [];
  const max = rows.length ? Math.max(...rows.map((r) => r.weight)) : 1;
  const log2n = Math.log2(Math.max(rows.length, 2));
  const headEntropy = chosen?.headEntropyBits ?? [];
  const meanEntropy =
    headEntropy.length > 0
      ? headEntropy.reduce((a, b) => a + b, 0) / headEntropy.length
      : null;

  return (
    <div data-testid="attention-panel">
      <div className="machine-label flex justify-between">
        <span>Attention / head-mean ← prior positions</span>
        {chosen && (
          <span className="text-ink tabular-nums">
            TOKEN {String(chosen.position).padStart(3, "0")} · L
            {String(chosen.layer).padStart(2, "0")}
          </span>
        )}
      </div>

      {!chosen ? (
        <p className="mt-3 text-muted">
          {traceMode && traceMode !== "RESEARCH"
            ? `Not collected — ${traceMode} mode. Run RESEARCH to measure attention.`
            : "Awaiting measurement…"}
        </p>
      ) : (
        <>
          <div className="mt-3 flex items-center gap-2">
            <span className="machine-label shrink-0">Layer</span>
            <div
              className="flex flex-wrap gap-x-2 gap-y-1"
              data-testid="attention-layers"
            >
              {attention!.map((a) => (
                <button
                  key={a.layer}
                  onClick={() => setLayerSel(a.layer)}
                  className={`w-7 border-b pb-0.5 font-mono text-machine tabular-nums transition-colors ${
                    a.layer === chosen.layer
                      ? "border-ink text-ink"
                      : "border-transparent text-muted hover:text-ink"
                  }`}
                >
                  {String(a.layer).padStart(2, "0")}
                </button>
              ))}
            </div>
          </div>

          <ul
            className="mt-3 max-h-64 space-y-1 overflow-y-auto pr-1"
            data-testid="attention-rows"
          >
            {rows.map((r) => (
              <li
                key={r.position}
                className="flex items-center gap-2"
                title={
                  r.position === -1
                    ? "position −1 — prepended BOS token (attention sink)"
                    : `position ${r.position}`
                }
              >
                <span className="w-14 shrink-0 truncate font-mono text-machine text-muted">
                  {r.position === -1 ? "<bos>" : r.text}
                </span>
                <div className="h-[3px] flex-1 bg-panel">
                  <div
                    className="h-full bg-ink/70"
                    style={{ width: `${(r.weight / max) * 100}%` }}
                  />
                </div>
                <span className="w-12 shrink-0 text-right font-mono text-machine tabular-nums text-muted">
                  {r.weight.toFixed(3)}
                </span>
              </li>
            ))}
          </ul>

          {headEntropy.length > 0 && meanEntropy !== null && (
            <div className="mt-4">
              <div className="machine-label flex justify-between">
                <span>Head entropy · full height = uniform (log₂n)</span>
                <span className="text-ink tabular-nums">
                  {meanEntropy.toFixed(2)} bits mean
                </span>
              </div>
              <div className="mt-1.5 flex h-4 items-end gap-px" aria-hidden>
                {headEntropy.map((h, i) => (
                  <div
                    key={i}
                    className="w-[3px] bg-ink/50"
                    style={{
                      height: `${Math.max((h / log2n) * 100, 8)}%`,
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
