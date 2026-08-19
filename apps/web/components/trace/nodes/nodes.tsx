import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { InputEvent, ConceptEvent, OutputEvent } from "@ets/trace-schema";
import { MiniBars } from "@/components/data-viz/MiniBars";
import { TraceNodeShell } from "./TraceNode";
import type { GenericNodeData, TokenNodeData } from "@/lib/trace/graph";

function InputNodeView({ data }: NodeProps<Node<GenericNodeData>>) {
  const event = data.event as InputEvent;
  return (
    <TraceNodeShell selected={data.selected} width={190}>
      <div className="machine-label">Input / measured</div>
      <p className="mt-1 line-clamp-2 font-serif text-sm leading-snug">
        {event.text}
      </p>
      <div className="machine-label mt-1.5">
        {event.tokenCount} tokens
      </div>
      <Handle type="source" position={Position.Right} className="!h-1.5 !w-1.5 !border-line !bg-paper" />
    </TraceNodeShell>
  );
}

function TokenNodeView({ data }: NodeProps<Node<TokenNodeData>>) {
  const event = data.event;
  const spike = event.entropyBits > 3.5;
  return (
    <TraceNodeShell selected={data.selected} latest={data.latest}>
      <Handle type="target" position={Position.Left} className="!h-1.5 !w-1.5 !border-line !bg-paper" />
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={`max-w-[110px] truncate font-mono text-sm ${
            data.latest ? "text-signal" : "text-ink"
          }`}
          title={event.rawText}
        >
          {event.leadingSpace ? "·" : ""}
          {event.text}
        </span>
        <span className="font-mono text-machine tabular-nums text-muted">
          {event.probability.toFixed(2)}
        </span>
      </div>
      <div className="machine-label mt-1 flex justify-between">
        <span>h {event.entropyBits.toFixed(2)}{spike ? " ▲" : ""}</span>
        <span>{event.latencyMs.toFixed(0)}ms</span>
      </div>
      <MiniBars layers={data.layers} />
      {data.concepts.length > 0 && (
        <div className="machine-label mt-1 italic">◆ {data.concepts.length} concept</div>
      )}
      <Handle type="source" position={Position.Right} className="!h-1.5 !w-1.5 !border-line !bg-paper" />
      {data.concepts.length > 0 && (
        <Handle
          id="concepts"
          type="source"
          position={Position.Bottom}
          className="!h-1.5 !w-1.5 !border-line !bg-paper"
        />
      )}
    </TraceNodeShell>
  );
}

function ConceptNodeView({ data }: NodeProps<Node<GenericNodeData>>) {
  const event = data.event as ConceptEvent;
  return (
    <TraceNodeShell selected={data.selected} width={160}>
      <Handle type="target" position={Position.Top} className="!h-1.5 !w-1.5 !border-line !bg-paper" />
      {/* INTERPRETED — visually distinct: muted, italic. Not a measurement. */}
      <div className="machine-label italic">Interpreted</div>
      <p className="mt-0.5 font-serif text-sm italic leading-snug text-muted">
        {event.label}
      </p>
      {/* the canvas shows each concept once — at the step where it peaked */}
      <div className="machine-label mt-1">peak {event.score.toFixed(2)}</div>
    </TraceNodeShell>
  );
}

function OutputNodeView({ data }: NodeProps<Node<GenericNodeData>>) {
  const event = data.event as OutputEvent;
  return (
    <TraceNodeShell selected={data.selected} width={190}>
      <Handle type="target" position={Position.Left} className="!h-1.5 !w-1.5 !border-line !bg-paper" />
      <div className="machine-label">Output / measured</div>
      <p className="mt-1 line-clamp-3 font-serif text-sm leading-snug">
        {event.text || "—"}
      </p>
      <div className="machine-label mt-1.5">
        {event.tokenCount} tokens · {event.durationMs}ms · {event.finishReason}
      </div>
    </TraceNodeShell>
  );
}

/** Fallback for event types with no dedicated node yet (DECISION, …). */
function GenericEventNodeView({ data }: NodeProps<Node<GenericNodeData>>) {
  return (
    <TraceNodeShell selected={data.selected} width={130}>
      <div className="machine-label">{data.event.type.toLowerCase()}</div>
      <div className="machine-label mt-1 text-ink">
        {data.event.level.toLowerCase()}
      </div>
    </TraceNodeShell>
  );
}

export {
  InputNodeView,
  TokenNodeView,
  ConceptNodeView,
  OutputNodeView,
  GenericEventNodeView,
};
