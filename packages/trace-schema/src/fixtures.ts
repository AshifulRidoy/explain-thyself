/**
 * Fixture loader — reads the committed fixture corpus from disk.
 * Used by the web app in fixture mode and by tests on both runtimes.
 */
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { validateTrace } from "./schema";
import type { Trace } from "./events";

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

export type FixtureKey = "trace-python-rust" | "trace-sky-blue" | "trace-minimal";

export async function loadFixture(key: FixtureKey): Promise<Trace> {
  const raw = await readFile(join(FIXTURES_DIR, `${key}.json`), "utf8");
  return validateTrace(JSON.parse(raw));
}

export async function listFixtures(): Promise<Trace[]> {
  const { FIXTURES } = await import("./generate");
  return Promise.all(FIXTURES.map((f) => loadFixture(f.key as FixtureKey)));
}
