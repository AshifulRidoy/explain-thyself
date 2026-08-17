import type { Metadata } from "next";
import { SIGNAL_TAXONOMY } from "@ets/trace-schema";

export const metadata: Metadata = {
  title: "Methodology",
  description:
    "What every signal in Explain The Self epistemically is: measured, derived, or interpreted.",
};

const LEVELS = ["MEASURED", "DERIVED", "INTERPRETED"] as const;

const LEVEL_INTRO: Record<(typeof LEVELS)[number], string> = {
  MEASURED:
    "Taken directly from the model run. If the interface shows it, the model produced it.",
  DERIVED:
    "Arithmetic on measurements — entropy, norms, ratios. Honest transformations, but transformations.",
  INTERPRETED:
    "A human or heuristic reading of signals. It may be wrong; it is never presented as measurement.",
};

export default function MethodologyPage() {
  const byLevel = LEVELS.map((level) => ({
    level,
    entries: Object.entries(SIGNAL_TAXONOMY).filter(
      ([, def]) => def.level === level,
    ),
  }));

  return (
    <div className="mx-auto max-w-6xl px-6 py-16">
      <p className="machine-label">Methodology</p>
      <h1 className="mt-4 max-w-3xl font-serif text-5xl leading-tight">
        This is an observability layer — not a claim about{" "}
        <span className="italic">the model&apos;s thoughts</span>.
      </h1>
      <p className="mt-6 max-w-2xl text-muted">
        Every quantity shown in the interface carries one of three labels.
        The label says what the number <em>is</em>, epistemically — before any
        story is told about it. This page is the complete registry; the
        Inspector badges come from the same source.
      </p>

      <div className="mt-16 grid gap-12 md:grid-cols-3">
        {byLevel.map(({ level, entries }) => (
          <section key={level}>
            <h2 className="font-mono text-sm uppercase tracking-widest">
              {level}
            </h2>
            <p className="mt-2 text-muted">{LEVEL_INTRO[level]}</p>
            <dl className="mt-6 divide-y divide-line border-t border-line">
              {entries.map(([key, def]) => (
                <div key={key} className="py-4">
                  <dt className="font-mono text-machine tracking-wide">
                    {key}
                  </dt>
                  <dd className="mt-1.5 text-muted">{def.definition}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </div>
  );
}
