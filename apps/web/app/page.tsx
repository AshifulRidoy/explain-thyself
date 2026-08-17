import Link from "next/link";
import { loadFixture } from "@ets/trace-schema/fixtures";
import { FixtureTeaser } from "@/components/trace/FixtureTeaser";
import { ExamineInput } from "@/components/shell/ExamineInput";

const LEVELS = [
  {
    n: "01",
    key: "MEASURED",
    line: "Taken directly from the model.",
    body: "Tokens, logits, hidden states — numbers the model actually produced, not summaries of them.",
  },
  {
    n: "02",
    key: "DERIVED",
    line: "Computed from measurements.",
    body: "Entropy, activation norms, ranks — arithmetic on top of what was measured, nothing more.",
  },
  {
    n: "03",
    key: "INTERPRETED",
    line: "A human reading of signals.",
    body: "Concept labels and hypotheses. Always marked as interpretation — never presented as what the model “thinks”.",
  },
] as const;

export default async function HomePage() {
  const fixture = await loadFixture("trace-sky-blue");
  const tokens = fixture.events
    .filter((e): e is Extract<typeof e, { type: "TOKEN" }> => e.type === "TOKEN")
    .slice(0, 48)
    .map((e) => ({
      text: e.leadingSpace ? ` ${e.text}` : e.text,
      entropyBits: e.entropyBits,
      probability: e.probability,
    }));

  return (
    <div>
      {/* ——— hero ——————————————————————————————————— */}
      <section className="mx-auto max-w-6xl px-6 pt-20 pb-16">
        <p className="machine-label">An instrument for examining language models</p>
        <h1 className="mt-6 font-serif text-hero leading-[0.95] tracking-tight">
          A microscope
          <br />
          for <span className="italic">artificial</span>
          <br />
          intelligence.
        </h1>
        <p className="mt-8 max-w-md text-base leading-relaxed text-muted">
          Watch a model generate, token by token — and see the measurable
          signals underneath the behavior: what it predicted, how uncertain
          it was, where inside the network things moved.
        </p>
        <div className="mt-10">
          <ExamineInput />
        </div>
      </section>

      {/* ——— instrument status ———————————————————— */}
      <section className="border-t border-line">
        <div className="mx-auto grid max-w-6xl grid-cols-2 gap-px px-6 sm:grid-cols-4">
          {[
            ["Model", "GPT-2 Small"],
            ["Layers", "12"],
            ["Parameters", "124M"],
            ["Trace Mode", "Standard"],
          ].map(([label, value]) => (
            <div key={label} className="py-6 sm:pr-6">
              <div className="machine-label">{label}</div>
              <div className="mt-2 font-mono text-sm uppercase tracking-wide">
                {value}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ——— a recorded trace ———————————————————— */}
      <section className="border-t border-line">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <div className="flex items-baseline justify-between">
            <h2 className="font-serif text-2xl">
              A trace, <span className="italic">recorded</span>
            </h2>
            <span className="machine-label">
              FIXTURE / TRACE-SKY-BLUE / DETERMINISTIC
            </span>
          </div>
          <div className="mt-8">
            <FixtureTeaser tokens={tokens} />
          </div>
          <p className="mt-6 max-w-md text-muted">
            Each bar is the entropy of the model&apos;s next-token distribution
            — its uncertainty, in bits. Spikes mark choice points; the accent
            follows the token being written.
          </p>
          <Link
            href="/explore?fixture=trace-sky-blue"
            className="machine-label mt-4 inline-block text-ink underline decoration-line underline-offset-4 transition-colors hover:text-signal"
          >
            Open this trace in the explorer →
          </Link>
        </div>
      </section>

      {/* ——— the epistemic ladder ——————————————————— */}
      <section className="border-t border-line">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <p className="machine-label">Every number is labeled by what it is</p>
          <div className="mt-8 grid gap-12 md:grid-cols-3">
            {LEVELS.map((lvl) => (
              <article key={lvl.key}>
                <div className="machine-label">{lvl.n}</div>
                <h3 className="mt-2 font-mono text-sm uppercase tracking-widest">
                  {lvl.key}
                </h3>
                <p className="mt-3 font-serif text-lg italic">{lvl.line}</p>
                <p className="mt-2 text-muted">{lvl.body}</p>
              </article>
            ))}
          </div>
          <Link
            href="/methodology"
            className="machine-label mt-10 inline-block text-ink underline decoration-line underline-offset-4 transition-colors hover:text-signal"
          >
            Read the methodology →
          </Link>
        </div>
      </section>
    </div>
  );
}
