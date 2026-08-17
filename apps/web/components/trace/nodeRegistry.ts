/**
 * THE spec §15 mapping: trace event `type` → canvas node component.
 * Adding an event type to the contract means adding one line here.
 */
import type { ComponentType } from "react";
import type { Node, NodeProps } from "@xyflow/react";
import {
  ConceptNodeView,
  GenericEventNodeView,
  InputNodeView,
  OutputNodeView,
  TokenNodeView,
} from "./nodes/nodes";
import type { EtsNodeData } from "@/lib/trace/graph";

type EtsNodeProps = NodeProps<Node<EtsNodeData>>;

export const nodeRegistry: Record<string, ComponentType<EtsNodeProps>> = {
  INPUT: InputNodeView as ComponentType<EtsNodeProps>,
  TOKEN: TokenNodeView as ComponentType<EtsNodeProps>,
  CONCEPT: ConceptNodeView as ComponentType<EtsNodeProps>,
  OUTPUT: OutputNodeView as ComponentType<EtsNodeProps>,
  // later phases render these with dedicated components; the fallback keeps
  // the canvas total over the contract
  DECISION: GenericEventNodeView as ComponentType<EtsNodeProps>,
  EVIDENCE: GenericEventNodeView as ComponentType<EtsNodeProps>,
  HYPOTHESIS: GenericEventNodeView as ComponentType<EtsNodeProps>,
  UNCERTAINTY: GenericEventNodeView as ComponentType<EtsNodeProps>,
};

export const NODE_TYPES = nodeRegistry;
