import { useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import type { TokenEvent } from "@ets/trace-schema";

/**
 * Current entropy + full-history sparkline, in bits. DERIVED quantity —
 * the badge in the Inspector says so; this meter only draws it.
 */
export function EntropyMeter({ tokens }: { tokens: TokenEvent[] }) {
  const { path, current, max } = useMemo(() => {
    if (tokens.length === 0) {
      return { path: null as string | null, current: null as number | null, max: 1 };
    }
    const bits = tokens.map((t) => t.entropyBits);
    const w = 220;
    const h = 36;
    const x = scaleLinear()
      .domain([0, Math.max(tokens.length - 1, 1)])
      .range([0, w]);
    const maxBits = Math.max(...bits, 1);
    const y = scaleLinear().domain([0, maxBits]).range([h, 2]);
    const gen = line<number>()
      .x((_, i) => x(i))
      .y((d) => y(d));
    return {
      path: gen(bits) ?? null,
      current: bits.at(-1) ?? null,
      max: maxBits,
    };
  }, [tokens]);

  return (
    <div>
      <div className="machine-label flex justify-between">
        <span>Entropy / bits</span>
        <span className="text-ink tabular-nums">
          {current === null ? "—" : current.toFixed(2)}
        </span>
      </div>
      <svg
        viewBox="0 0 220 36"
        className="mt-2 h-9 w-full border-b border-line"
        preserveAspectRatio="none"
      >
        {path && (
          <path
            d={path}
            fill="none"
            stroke="var(--ink)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>
      <div className="machine-label mt-1 flex justify-between">
        <span>0</span>
        <span>{max.toFixed(1)}</span>
      </div>
    </div>
  );
}
