"use client";

import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

export function ExamineInput() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const q = prompt.trim();
    if (!q) return;
    router.push(`/explore?prompt=${encodeURIComponent(q)}`);
  };

  return (
    <form onSubmit={submit} className="w-full max-w-xl">
      <label htmlFor="examine" className="machine-label block">
        What would you like to examine?
      </label>
      <div className="mt-3 flex items-baseline gap-4 border-b border-ink pb-2 focus-within:border-signal">
        <input
          id="examine"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder='e.g. "Why is the sky blue?"'
          className="w-full bg-transparent font-serif text-xl outline-none placeholder:text-muted/60"
          autoComplete="off"
        />
        <button
          type="submit"
          className="machine-label shrink-0 text-ink transition-colors hover:text-signal"
        >
          Examine →
        </button>
      </div>
    </form>
  );
}
