"use client";

import dynamic from "next/dynamic";
import { ReactFlowProvider } from "@xyflow/react";
import { useMemo } from "react";
import type { Trace } from "@ets/trace-schema";
import { useTraceStore, tokenEvents, layerActivityByPosition, attentionByPosition, conceptsTimeline, conceptsByPosition, type DataSourceMode } from "@/lib/trace/store";
import { useFixtureReplay } from "@/lib/trace/useTraceDataSource";
import { TraceHeader } from "./TraceHeader";
import { PlaybackControls } from "./PlaybackControls";
import { Inspector } from "./Inspector";
import { TokenStream } from "./TokenStream";
import { EntropyMeter } from "@/components/data-viz/EntropyMeter";
import { ConceptPanel } from "@/components/data-viz/ConceptPanel";
import { UncertaintyPanel } from "@/components/data-viz/UncertaintyPanel";
import { uncertaintyQuantities } from "@/lib/trace/uncertainty";

const TraceCanvas = dynamic(
  () => import("./TraceCanvas").then((m) => m.TraceCanvas),
  { ssr: false },
);

/**
 * The instrument: canvas | inspector over a full-width token stream.
 * Desktop grid; mobile stacks (spec §34) — never squeezed columns.
 * Plays any complete Trace — a committed fixture or a saved trace
 * loaded back from Postgres (mode="replay").
 */
export function ExploreClient({
  trace,
  mode = "fixture",
}: {
  trace: Trace;
  mode?: DataSourceMode;
}) {
  const controls = useFixtureReplay(trace, mode);
  const events = useTraceStore((s) => s.events);
  const envelope = useTraceStore((s) => s.envelope);
  const status = useTraceStore((s) => s.status);
  const error = useTraceStore((s) => s.error);
  const selectedEventId = useTraceStore((s) => s.selectedEventId);
  const select = useTraceStore((s) => s.select);

  const tokens = useMemo(() => tokenEvents(events), [events]);
  const layersByPos = useMemo(() => layerActivityByPosition(events), [events]);
  const attentionByPos = useMemo(() => attentionByPosition(events), [events]);
  const conceptTimeline = useMemo(() => conceptsTimeline(events), [events]);
  const conceptsByPos = useMemo(() => conceptsByPosition(events), [events]);
  const uncertainty = useMemo(() => uncertaintyQuantities(events), [events]);
  const selected = useMemo(
    () => events.find((e) => e.id === selectedEventId) ?? null,
    [events, selectedEventId],
  );
  const selectedLayers =
    selected?.type === "TOKEN" ? layersByPos.get(selected.position) ?? null : null;
  const selectedAttention =
    selected?.type === "TOKEN" ? attentionByPos.get(selected.position) ?? null : null;
  const selectedConcepts =
    selected?.type === "TOKEN" ? conceptsByPos.get(selected.position) ?? null : null;

  return (
    <div>
      <TraceHeader
        envelope={envelope ?? trace}
        tokenCount={tokens.length}
        status={status}
        mode={mode}
      />

      {status === "error" && (
        <p className="machine-label px-6 py-3 text-signal">
          Stream failed — {error}
        </p>
      )}

      <PlaybackControls controls={controls} />

      <div className="mx-auto grid max-w-6xl gap-0 px-6 lg:grid-cols-[1fr_340px]">
        <div className="h-[520px] border border-line lg:border-0 lg:border-r">
          <ReactFlowProvider>
            <TraceCanvas />
          </ReactFlowProvider>
        </div>
        <aside className="space-y-8 px-0 py-6 lg:px-6">
          <Inspector
            event={selected}
            layerActivity={selectedLayers}
            attention={selectedAttention}
            concepts={selectedConcepts}
            traceMode={(envelope ?? trace).traceMode}
          />
          <div className="border-t border-line pt-6">
            <ConceptPanel
              activities={conceptTimeline}
              tokenCount={tokens.length}
              traceMode={(envelope ?? trace).traceMode}
            />
          </div>
          <div className="border-t border-line pt-6">
            <UncertaintyPanel
              quantities={uncertainty}
              traceMode={(envelope ?? trace).traceMode}
              complete={status === "complete"}
            />
          </div>
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
