"use client";

import { useEffect, useMemo, useRef, useState } from "react";

export interface TeaserToken {
  text: string;
  entropyBits: number;
  probability: number;
}

/**
 * Looping replay of a recorded trace: tokens surface one at a time, each
 * carrying a hairline bar for its entropy. Signal marks the two things that
 * deserve attention — the token that just appeared, and entropy spikes.
 */
export function FixtureTeaser({ tokens }: { tokens: TeaserToken[] }) {
  const [count, setCount] = useState(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const spikeThreshold = useMemo(() => {
    if (tokens.length === 0) return Infinity;
    const bits = tokens.map((t) => t.entropyBits);
    const mean = bits.reduce((a, b) => a + b, 0) / bits.length;
    const variance =
      bits.reduce((a, b) => a + (b - mean) ** 2, 0) / bits.length;
    return mean + Math.sqrt(variance);
  }, [tokens]);

  const maxBits = useMemo(
    () => Math.max(...tokens.map((t) => t.entropyBits), 1),
    [tokens],
  );

  useEffect(() => {
    const stepMs = 170;
    const holdMs = 2600;
    let cancelled = false;

    const tick = () => {
      if (cancelled) return;
      setCount((c) => {
        if (c >= tokens.length) {
          timer.current = setTimeout(() => !cancelled && setCount(0), holdMs);
          return c;
        }
        return c + 1;
      });
      timer.current = setTimeout(tick, stepMs);
    };
    tick();
    return () => {
      cancelled = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [tokens]);

  const current = count > 0 ? tokens[count - 1] : null;

  return (
    <figure className="w-full">
      <div className="flex items-end gap-[3px] overflow-hidden border-b border-line pb-2">
        {tokens.slice(0, count).map((token, i) => {
          const isLatest = i === count - 1;
          const isSpike = token.entropyBits >= spikeThreshold;
          const barColor = isSpike || isLatest ? "bg-signal" : "bg-ink";
          return (
            <div
              key={i}
              className="flex w-auto min-w-[10px] flex-col items-center gap-1.5"
            >
              <div className="flex h-12 w-full items-end">
                <div
                  className={`w-[3px] transition-all duration-200 ${barColor}`}
                  style={{
                    height: `${Math.max((token.entropyBits / maxBits) * 100, 4)}%`,
                    opacity: isSpike ? 1 : 0.55,
                  }}
                />
              </div>
              <span
                className={`font-mono text-machine leading-none ${
                  isLatest ? "text-signal" : "text-ink/80"
                }`}
              >
                {token.text}
              </span>
            </div>
          );
        })}
      </div>
      <figcaption className="machine-label mt-3 flex h-4 items-center gap-6 tabular-nums">
        <span>
          TOKEN {String(Math.max(count - 1, 0)).padStart(3, "0")} /{" "}
          {String(tokens.length - 1).padStart(3, "0")}
        </span>
        {current && (
          <>
            <span>ENTROPY {current.entropyBits.toFixed(2)} BITS</span>
            <span>P {current.probability.toFixed(2)}</span>
          </>
        )}
      </figcaption>
    </figure>
  );
}
