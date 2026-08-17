"use client";

import { useCallback, useEffect, useMemo } from "react";
import {
  Background,
  ReactFlow,
  useReactFlow,
  type NodeMouseHandler,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useTraceStore } from "@/lib/trace/store";
import { toGraph } from "@/lib/trace/graph";
import { nodeRegistry } from "./nodeRegistry";

/**
 * The canvas. Client-only (mounted via dynamic import with ssr:false).
 * Nodes are never dragged or animated into place — data moves, chrome
 * doesn't.
 */
export function TraceCanvas() {
  const events = useTraceStore((s) => s.events);
  const selectedEventId = useTraceStore((s) => s.selectedEventId);
  const status = useTraceStore((s) => s.status);
  const select = useTraceStore((s) => s.select);

  const { nodes, edges } = useMemo(
    () => toGraph(events, selectedEventId, status === "streaming"),
    [events, selectedEventId, status],
  );

  const { fitView } = useReactFlow();

  // keep the newest part of the trace in view while streaming
  const lastId = events.at(-1)?.id ?? null;
  useEffect(() => {
    if (status === "streaming" && lastId) {
      void fitView({ nodes: [{ id: lastId }], duration: 300, maxZoom: 1.1 });
    }
  }, [lastId, status, fitView]);

  const onNodeClick = useCallback<NodeMouseHandler>(
    (_, node) => select(node.id),
    [select],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeRegistry}
      onNodeClick={onNodeClick}
      panOnScroll
      zoomOnScroll={false}
      nodesConnectable={false}
      elementsSelectable
      proOptions={{ hideAttribution: true }}
      colorMode="light"
    >
      <Background gap={24} color="var(--line)" lineWidth={1} />
    </ReactFlow>
  );
}
