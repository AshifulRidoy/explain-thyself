import type { LayerActivityEvent } from "@ets/trace-schema";

/**
 * The 12-bar layer strip inside a TokenNode — the node IS the measurement.
 * Bars are normRatio (activity vs its running mean); Signal marks >1.
 */
export function MiniBars({ layers }: { layers: LayerActivityEvent | null }) {
  if (!layers) return null;
  const max = Math.max(...layers.layers.map((l) => l.normRatio), 1.4);
  return (
    <div className="mt-1.5 flex h-4 items-end gap-px" aria-hidden>
      {layers.layers.map((l) => {
        const above = l.normRatio > 1;
        return (
          <div
            key={l.layer}
            className={`w-[3px] ${above ? "bg-signal" : "bg-ink/50"}`}
            style={{ height: `${Math.max((l.normRatio / max) * 100, 10)}%` }}
          />
        );
      })}
    </div>
  );
}
