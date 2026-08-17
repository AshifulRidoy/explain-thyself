"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

interface Command {
  id: string;
  label: string;
  hint: string;
  perform: (router: ReturnType<typeof useRouter>) => void;
}

const COMMANDS: Command[] = [
  {
    id: "fixture:python-rust",
    label: "Open fixture — Python vs Rust",
    hint: "explore",
    perform: (r) => r.push("/explore?fixture=trace-python-rust"),
  },
  {
    id: "fixture:sky-blue",
    label: "Open fixture — Why is the sky blue?",
    hint: "explore",
    perform: (r) => r.push("/explore?fixture=trace-sky-blue"),
  },
  {
    id: "fixture:minimal",
    label: "Open fixture — minimal (3 tokens)",
    hint: "explore",
    perform: (r) => r.push("/explore?fixture=trace-minimal"),
  },
  {
    id: "page:home",
    label: "Home",
    hint: "page",
    perform: (r) => r.push("/"),
  },
  {
    id: "page:traces",
    label: "Saved traces",
    hint: "page",
    perform: (r) => r.push("/traces"),
  },
  {
    id: "page:methodology",
    label: "Methodology",
    hint: "page",
    perform: (r) => r.push("/methodology"),
  },
];

export function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
        setQuery("");
        setActive(0);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return COMMANDS;
    return COMMANDS.filter(
      (c) =>
        c.label.toLowerCase().includes(q) || c.hint.toLowerCase().includes(q),
    );
  }, [query]);

  if (!open) return null;

  const run = (cmd: Command) => {
    setOpen(false);
    cmd.perform(router);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-ink/20 pt-28"
      onClick={() => setOpen(false)}
    >
      <div
        className="w-full max-w-md border border-ink bg-paper shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown")
              setActive((a) => Math.min(a + 1, results.length - 1));
            else if (e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
            else if (e.key === "Enter" && results[active]) run(results[active]);
          }}
          placeholder="Search commands…"
          className="w-full border-b border-line bg-transparent px-4 py-3 font-mono text-sm outline-none placeholder:text-muted"
        />
        <ul className="max-h-72 overflow-y-auto py-1">
          {results.map((cmd, i) => (
            <li key={cmd.id}>
              <button
                onClick={() => run(cmd)}
                onMouseEnter={() => setActive(i)}
                className={`flex w-full items-center justify-between px-4 py-2.5 text-left text-interface ${
                  i === active ? "bg-panel" : ""
                }`}
              >
                <span>{cmd.label}</span>
                <span className="machine-label">{cmd.hint}</span>
              </button>
            </li>
          ))}
          {results.length === 0 && (
            <li className="machine-label px-4 py-3">No matching command</li>
          )}
        </ul>
      </div>
    </div>
  );
}
