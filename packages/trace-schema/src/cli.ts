/**
 * `pnpm fixture` — (re)generate the committed fixture corpus.
 * Deterministic: same seeds ⇒ byte-identical files.
 */
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { FIXTURES, generateTraceFixture } from "./generate.js";
import { traceSchema } from "./schema.js";

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "fixtures");

async function main() {
  await mkdir(FIXTURES_DIR, { recursive: true });
  for (const spec of FIXTURES) {
    const trace = generateTraceFixture(spec);
    // the generator itself must emit schema-valid output — fail here, not in the browser
    traceSchema.parse(trace);
    const path = join(FIXTURES_DIR, `${spec.key}.json`);
    await writeFile(path, JSON.stringify(trace, null, 2) + "\n", "utf8");
    const tokens = trace.events.filter((e) => e.type === "TOKEN").length;
    console.log(`${spec.key}.json  events=${trace.events.length} tokens=${tokens}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
