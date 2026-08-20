import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ExploreClient } from "@/components/trace/ExploreClient";
import { listSimilarTraces, loadTraceById, type SimilarTrace } from "@/lib/trace/repo";

export const dynamic = "force-dynamic"; // replay must reflect final DB state

interface PageProps {
  params: Promise<{ id: string }>;
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { id } = await params;
  return { title: `Trace ${id}` };
}

const REASON_COPY: Record<string, string> = {
  invalid:
    "This trace's stored events no longer satisfy the trace contract — the schema evolved past the data. Regenerate the trace.",
  db: "Postgres is unreachable. Is the database up? (`docker-compose up -d`)",
};

export default async function TraceReplayPage({ params }: PageProps) {
  const { id } = await params;
  const result = await loadTraceById(id);

  if (!result.ok) {
    if (result.reason === "not_found") notFound();
    return (
      <div className="mx-auto max-w-2xl px-6 py-24">
        <p className="machine-label text-signal">Replay unavailable</p>
        <p className="mt-3 text-muted">{REASON_COPY[result.reason]}</p>
        <Link
          href="/traces"
          className="machine-label mt-8 inline-block text-ink underline decoration-line underline-offset-4 hover:text-signal"
        >
          ← All traces
        </Link>
      </div>
    );
  }

  const { trace } = result;
  // spec §28: rank against the trace's own stored embedding (Drizzle read,
  // engine-independent). null/[] render nothing — never a fake ranking.
  const similar = await listSimilarTraces(trace.id, 5);

  return (
    <div>
      <div className="border-b border-line">
        <div className="mx-auto flex max-w-6xl items-center gap-4 px-6 py-3">
          <span className="machine-label">
            Saved trace — replaying recorded events
          </span>
          <Link
            href="/traces"
            className="machine-label ml-auto text-muted transition-colors hover:text-signal"
          >
            ← All traces
          </Link>
        </div>
      </div>
      <ExploreClient key={trace.id} trace={trace} mode="replay" />
      {similar && similar.length > 0 && <SimilarTracesSection hits={similar} />}
    </div>
  );
}

function SimilarTracesSection({ hits }: { hits: SimilarTrace[] }) {
  return (
    <section className="border-t border-line">
      <div className="mx-auto max-w-6xl px-6 py-8">
        <h2 className="font-serif text-2xl leading-tight">
          Traces the model represents similarly
        </h2>
        <p className="machine-label mt-1 text-muted">
          Spec §28 · DERIVED — cosine of the final-layer prompt
          representation · rank, not meaning
        </p>
        <ol className="mt-4 border-t border-line">
          {hits.map((hit) => (
            <li key={hit.traceId} className="border-b border-line">
              <Link
                href={`/trace/${hit.traceId}`}
                className="grid grid-cols-[5rem_1fr_6rem] items-baseline gap-x-4 px-2 py-3 transition-colors hover:bg-panel"
              >
                <span className="font-mono text-xs tabular-nums text-muted">
                  {String(hit.displayId).padStart(4, "0")}
                </span>
                <span className="truncate font-serif text-lg italic">
                  {hit.input.length > 72 ? `${hit.input.slice(0, 71)}…` : hit.input}
                </span>
                <span className="text-right font-mono text-xs tabular-nums text-muted">
                  {hit.similarity >= 0 ? "+" : ""}
                  {hit.similarity.toFixed(4)}
                </span>
              </Link>
            </li>
          ))}
        </ol>
        <p className="machine-label mt-4 text-muted">
          GPT-2&apos;s hidden space is anisotropic — absolute cosines are
          compressed; open both replays to compare what the model actually
          did.
        </p>
      </div>
    </section>
  );
}
