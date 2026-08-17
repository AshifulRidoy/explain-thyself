import type { TopToken } from "@ets/trace-schema";

/**
 * Top-k next-token distribution: mono rows, hairline weight bars.
 * MEASURED — softmax of the logits, nothing added.
 */
export function ProbabilityDistribution({
  topK,
  sampledTokenId,
}: {
  topK: TopToken[];
  sampledTokenId: number;
}) {
  const max = Math.max(...topK.map((t) => t.probability), 1e-9);
  return (
    <div>
      <div className="machine-label">Next-token distribution / top 8</div>
      <ul className="mt-3 space-y-1.5">
        {topK.map((tok) => {
          const sampled = tok.tokenId === sampledTokenId;
          return (
            <li key={tok.tokenId} className="flex items-center gap-3">
              <span
                className={`w-28 shrink-0 truncate font-mono text-machine ${
                  sampled ? "text-signal" : "text-ink/80"
                }`}
                title={tok.rawText}
              >
                {tok.leadingSpace ? "·" : ""}
                {tok.text}
              </span>
              <div className="h-[3px] flex-1 bg-panel">
                <div
                  className={`h-full ${sampled ? "bg-signal" : "bg-ink/70"}`}
                  style={{ width: `${(tok.probability / max) * 100}%` }}
                />
              </div>
              <span className="w-12 shrink-0 text-right font-mono text-machine tabular-nums text-muted">
                {tok.probability.toFixed(3)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
