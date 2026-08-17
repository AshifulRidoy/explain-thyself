import type { Trace } from "@ets/trace-schema";
import type { DataSourceMode, StreamStatus } from "@/lib/trace/store";

const MODE_LABEL: Record<DataSourceMode, string> = {
  fixture: "Fixture",
  live: "Live",
  replay: "Replay",
};

const STATUS_LABEL: Record<StreamStatus, string> = {
  idle: "Idle",
  streaming: "Streaming",
  complete: "Complete",
  error: "Error",
};

/** Mono status row — the instrument's front panel. */
export function TraceHeader({
  envelope,
  tokenCount,
  status,
  mode,
}: {
  envelope: Trace | null;
  tokenCount: number;
  status: StreamStatus;
  mode: DataSourceMode;
}) {
  return (
    <div className="border-b border-line">
      <dl className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-8 gap-y-2 px-6 py-3">
        <div className="flex items-baseline gap-2">
          <dt className="machine-label">Trace</dt>
          <dd className="font-mono text-sm uppercase">
            {envelope
              ? String(envelope.displayId).padStart(4, "0")
              : "—"}
          </dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="machine-label">Model</dt>
          <dd className="font-mono text-sm uppercase">{envelope?.model.name ?? "—"}</dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="machine-label">Mode</dt>
          <dd className="font-mono text-sm uppercase">{MODE_LABEL[mode]}</dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="machine-label">Tokens</dt>
          <dd className="font-mono text-sm tabular-nums">
            {String(tokenCount).padStart(3, "0")}
          </dd>
        </div>
        <div className="flex items-baseline gap-2">
          <dt className="machine-label">Status</dt>
          <dd
            className={`font-mono text-sm uppercase ${
              status === "streaming" ? "text-signal" : status === "error" ? "text-signal" : ""
            }`}
          >
            {STATUS_LABEL[status]}
          </dd>
        </div>
      </dl>
    </div>
  );
}
