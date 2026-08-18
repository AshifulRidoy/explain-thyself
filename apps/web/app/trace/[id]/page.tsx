import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ExploreClient } from "@/components/trace/ExploreClient";
import { loadTraceById } from "@/lib/trace/repo";

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
      <ExploreClient trace={trace} mode="replay" />
    </div>
  );
}
