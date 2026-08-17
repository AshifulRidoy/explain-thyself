import type { ReactNode } from "react";

/**
 * Shared node chrome: a physically distinct object on the canvas — the one
 * place the design system allows a box. Selection is a Signal ring;
 * "something deserves attention" and nothing else.
 */
export function TraceNodeShell({
  selected,
  latest,
  children,
  width = 150,
}: {
  selected: boolean;
  latest?: boolean;
  children: ReactNode;
  width?: number;
}) {
  return (
    <div
      className={`rounded-[2px] border bg-paper px-2.5 py-2 transition-shadow ${
        selected ? "border-signal shadow-[0_0_0_3px_var(--signal)]" : "border-line"
      }`}
      style={{ width }}
    >
      {children}
      {latest && (
        <div className="machine-label mt-1 text-signal">● live</div>
      )}
    </div>
  );
}
