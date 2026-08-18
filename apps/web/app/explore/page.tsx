import type { Metadata } from "next";
import { loadFixture, type FixtureKey } from "@ets/trace-schema/fixtures";
import { ExploreClient } from "@/components/trace/ExploreClient";
import { LiveExploreClient } from "@/components/trace/LiveExploreClient";

export const metadata: Metadata = {
  title: "Explore",
  description: "Open a trace — fixture, live generation, or saved replay.",
};

const DEFAULT_FIXTURE: FixtureKey = "trace-python-rust";

function parseFixtureKey(value: string | undefined): FixtureKey | null {
  if (!value) return null;
  const withPrefix = value.startsWith("trace-") ? value : `trace-${value}`;
  return (
    (["trace-python-rust", "trace-sky-blue", "trace-minimal"] as const).includes(
      withPrefix as FixtureKey,
    )
      ? (withPrefix as FixtureKey)
      : null
  );
}

export default async function ExplorePage({
  searchParams,
}: {
  searchParams: Promise<{ fixture?: string; prompt?: string }>;
}) {
  const params = await searchParams;
  const prompt = params.prompt?.trim();
  if (prompt) {
    return <LiveExploreClient prompt={prompt} />;
  }
  const key = parseFixtureKey(params.fixture) ?? DEFAULT_FIXTURE;
  const trace = await loadFixture(key);
  return <ExploreClient trace={trace} />;
}
