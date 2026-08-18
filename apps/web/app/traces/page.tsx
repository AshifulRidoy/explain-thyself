import type { Metadata } from "next";
import Link from "next/link";
import type { TraceSummary } from "@/lib/trace/repo";
import { listRecentTraces } from "@/lib/trace/repo";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Traces",
  description: "Recently recorded traces — every one replays event-for-event.",
};

function formatDuration(ms: number | null): string {
  if (ms === null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function formatWhen(createdAt: Date): string {
  // fixed columns, mono — an instrument's log, not a social feed
  const iso = createdAt.toISOString();
  return `${iso.slice(0, 10)} ${iso.slice(11, 19)}`;
}

function excerpt(text: string, max = 64): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

export default async function TracesPage() {
  let rows: TraceSummary[];
  let dbDown = false;
  try {
    rows = await listRecentTraces(50);
  } catch {
    rows = [];
    dbDown = true;
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-12">
      <header className="mb-10">
        <h1 className="font-serif text-4xl leading-tight">Recorded traces</h1>
        <p className="mt-2 max-w-prose text-muted">
          Every live generation is persisted with its events and replays
          exactly as it streamed — same probabilities, same entropy, same
          timing.
        </p>
      </header>

      {dbDown && (
        <p className="machine-label text-signal">
          Postgres unreachable — run `docker-compose up -d` and reload.
        </p>
      )}

      {!dbDown && rows.length === 0 && (
        <div className="border-y border-line py-16">
          <p className="font-serif text-xl italic text-muted">
            Nothing recorded yet.
          </p>
          <Link
            href="/explore"
            className="machine-label mt-4 inline-block text-ink underline decoration-line underline-offset-4 hover:text-signal"
          >
            Run the first trace →
          </Link>
        </div>
      )}

      {rows.length > 0 && (
        <ol className="border-t border-line">
          {rows.map((row) => (
            <li key={row.id} className="border-b border-line">
              <Link
                href={`/trace/${row.id}`}
                className="grid grid-cols-[auto_1fr_auto] items-baseline gap-x-4 px-2 py-3 transition-colors hover:bg-panel sm:grid-cols-[5rem_1fr_7rem_5rem_6rem]"
              >
                <span className="font-mono text-xs tabular-nums text-muted">
                  {String(row.displayId).padStart(4, "0")}
                </span>
                <span className="truncate font-serif text-lg italic">
                  {excerpt(row.input)}
                </span>
                <span className="hidden font-mono text-xs uppercase text-muted sm:block">
                  {row.modelName} · {row.traceMode.toLowerCase()}
                </span>
                <span
                  className={`hidden text-right font-mono text-xs uppercase sm:block ${
                    row.status === "error" ? "text-signal" : "text-muted"
                  }`}
                >
                  {row.status}
                </span>
                <span className="text-right font-mono text-xs tabular-nums text-muted">
                  {row.tokenCount ?? 0} tok · {formatDuration(row.durationMs)}
                </span>
              </Link>
            </li>
          ))}
        </ol>
      )}

      {rows.length > 0 && (
        <p className="machine-label mt-6 text-muted">
          Showing the {rows.length} most recent · timestamps{" "}
          {formatWhen(rows[0].createdAt)} → {formatWhen(rows[rows.length - 1].createdAt)} UTC
        </p>
      )}
    </div>
  );
}
