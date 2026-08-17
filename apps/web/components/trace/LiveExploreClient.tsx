"use client";

import dynamic from "next/dynamic";
import { ReactFlowProvider } from "@xyflow/react";
import { useEffect, useMemo } from "react";
import { useTraceStore, tokenEvents, layerActivityByPosition } from "@/lib/trace/store";
import { useLiveTrace } from "@/lib/trace/useTraceDataSource";
import { TraceHeader } from "./TraceHeader";
import { Inspector } from "./Inspector";
import { TokenStream } from "./TokenStream";
import { EntropyMeter } from "@/components/data-viz/EntropyMeter";

const TraceCanvas = dynamic(
  () => import("./TraceCanvas").then((m) => m.TraceCanvas),
  { ssr: false },
);

const MAX_TOKENS = 30;

/**
 * Live mode: the same instrument, fed by the trace engine over SSE.
 * The only difference from fixture mode is the data source — the reducer,
 * canvas, inspector and stream are shared down to the component.
 */
export function LiveExploreClient({ prompt }: { prompt: string }) {
  const { running, start, stop } = useLiveTrace();
  const events = useTraceStore((s) => s.events);
  const envelope = useTraceStore((s) => s.envelope);
  const status = useTraceStore((s) => s.status);
  const error = useTraceStore((s) => s.error);
  const selectedEventId = useTraceStore((s) => s.selectedEventId);
  const select = useTraceStore((s) => s.select);

  useEffect(() => {
    void start({ prompt, maxTokens: MAX_TOKENS, traceMode: "BASIC" });
    return stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prompt]);

  const tokens = useMemo(() => tokenEvents(events), [events]);
  const layersByPos = useMemo(() => layerActivityByPosition(events), [events]);
  const selected = useMemo(
    () => events.find((e) => e.id === selectedEventId) ?? null,
    [events, selectedEventId],
  );
  const selectedLayers =
    selected?.type === "TOKEN" ? layersByPos.get(selected.position) ?? null : null;

  return (
    <div>
      <TraceHeader
        envelope={envelope}
        tokenCount={tokens.length}
        status={status}
        mode="live"
      />

      {status === "error" && (
        <p className="machine-label px-6 py-3 text-signal">
          Stream failed — {error}. Is the engine running on :8000?
        </p>
      )}

      <div className="flex items-center gap-4 border-b border-line px-6 py-3">
        <span className="machine-label">Prompt</span>
        <span className="truncate font-serif text-base italic">{prompt}</span>
        {running ? (
          <button
            onClick={stop}
            className="machine-label ml-auto shrink-0 text-signal"
          >
            ■ Stop
          </button>
        ) : (
          <button
            onClick={() => void start({ prompt, maxTokens: MAX_TOKENS, traceMode: "BASIC" })}
            className="machine-label ml-auto shrink-0 text-ink transition-colors hover:text-signal"
          >
            ↻ Re-run
          </button>
        )}
      </div>

      <div className="mx-auto grid max-w-6xl gap-0 px-6 lg:grid-cols-[1fr_340px]">
        <div className="h-[520px] border border-line lg:border-0 lg:border-r">
          <ReactFlowProvider>
            <TraceCanvas />
          </ReactFlowProvider>
        </div>
        <aside className="space-y-8 px-0 py-6 lg:px-6">
          <Inspector event={selected} layerActivity={selectedLayers} />
          <div className="border-t border-line pt-6">
            <EntropyMeter tokens={tokens} />
          </div>
        </aside>
      </div>

      <div className="border-t border-line">
        <div className="machine-label px-6 pt-3">Token stream</div>
        <TokenStream
          tokens={tokens}
          selectedId={selectedEventId}
          streaming={status === "streaming"}
          onSelect={select}
        />
      </div>
    </div>
  );
}
