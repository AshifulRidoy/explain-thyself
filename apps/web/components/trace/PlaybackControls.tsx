import type { DataSourceControls } from "@/lib/trace/useTraceDataSource";

const SPEEDS = [1, 4, 20] as const;

/** Playback controls for replayed data sources (fixture / saved trace). */
export function PlaybackControls({ controls }: { controls: DataSourceControls }) {
  return (
    <div className="flex items-center gap-4 px-6 py-3">
      <button
        onClick={controls.playing ? controls.pause : controls.play}
        className="machine-label text-ink transition-colors hover:text-signal"
      >
        {controls.playing ? "❚❚ Pause" : "▶ Play"}
      </button>
      <button
        onClick={controls.reset}
        className="machine-label text-ink transition-colors hover:text-signal"
      >
        ↺ Reset
      </button>
      <div className="ml-auto flex items-center gap-2">
        <span className="machine-label">Speed</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => controls.setSpeed(s)}
            className={`font-mono text-machine transition-colors ${
              controls.speed === s ? "text-signal" : "text-muted hover:text-ink"
            }`}
          >
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
