import type { TokenEvent } from "@ets/trace-schema";

/**
 * Full-width token stream under the canvas — the trace as text, in order.
 * Click a token to select it; the accent marks selection and entropy spikes.
 */
export function TokenStream({
  tokens,
  selectedId,
  streaming,
  onSelect,
}: {
  tokens: TokenEvent[];
  selectedId: string | null;
  streaming: boolean;
  onSelect: (id: string) => void;
}) {
  if (tokens.length === 0) {
    return <p className="machine-label px-6 py-4">Awaiting first token…</p>;
  }
  const bits = tokens.map((t) => t.entropyBits);
  const mean = bits.reduce((a, b) => a + b, 0) / bits.length;
  const variance = bits.reduce((a, b) => a + (b - mean) ** 2, 0) / bits.length;
  const spikeAt = mean + Math.sqrt(variance);

  return (
    <div className="flex flex-wrap gap-x-1 gap-y-2 px-6 py-4 font-mono text-sm leading-relaxed">
      {tokens.map((tok) => {
        const selected = tok.id === selectedId;
        const isSpike = tok.entropyBits >= spikeAt;
        return (
          <button
            key={tok.id}
            onClick={() => onSelect(tok.id)}
            title={`p ${tok.probability.toFixed(3)} · h ${tok.entropyBits.toFixed(2)} bits`}
            className={`cursor-pointer px-0.5 underline-offset-4 transition-colors ${
              selected
                ? "bg-signal text-paper"
                : isSpike
                  ? "text-signal hover:bg-panel"
                  : "text-ink hover:bg-panel"
            }`}
          >
            {tok.leadingSpace ? " " : ""}
            {tok.text}
            {streaming && tok === tokens.at(-1) && (
              <span className="ml-0.5 animate-pulse">▊</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
