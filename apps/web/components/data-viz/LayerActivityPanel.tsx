import type { LayerActivityEvent } from "@ets/trace-schema";

/**
 * L01…L12 residual-norm rows for one position. DERIVED — ‖resid_post‖₂ at
 * the final position, ratio vs the layer's running mean across steps.
 */
export function LayerActivityPanel({
  activity,
}: {
  activity: LayerActivityEvent | null;
}) {
  return (
    <div>
      <div className="machine-label flex justify-between">
        <span>Layer activity / ‖resid post‖₂</span>
        {activity && (
          <span className="text-ink tabular-nums">
            TOKEN {String(activity.position).padStart(3, "0")}
          </span>
        )}
      </div>
      {!activity ? (
        <p className="mt-3 text-muted">Awaiting measurement…</p>
      ) : (
        <ul className="mt-3 space-y-1">
          {[...activity.layers].reverse().map((l) => {
            const above = l.normRatio > 1;
            return (
              <li key={l.layer} className="flex items-center gap-3">
                <span className="w-8 shrink-0 font-mono text-machine text-muted">
                  L{String(l.layer).padStart(2, "0")}
                </span>
                <div className="h-[3px] flex-1 bg-panel">
                  <div
                    className={`h-full ${above ? "bg-signal" : "bg-ink/70"}`}
                    style={{
                      width: `${Math.min((l.normRatio / 1.6) * 100, 100)}%`,
                    }}
                  />
                </div>
                <span className="w-14 shrink-0 text-right font-mono text-machine tabular-nums text-muted">
                  {l.l2Norm.toFixed(1)}
                </span>
                <span className="w-10 shrink-0 text-right font-mono text-machine tabular-nums">
                  ×{l.normRatio.toFixed(2)}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
