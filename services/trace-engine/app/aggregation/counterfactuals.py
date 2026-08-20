"""Counterfactual mode (spec §23 / Phase 6): what would change the answer?

A *variable* is one word-level substitution from the human-authored
dictionary below (spec §26 would use an LLM app layer to generate
counterfactual questions; this instrument has none, so — like the concept
dictionary — the variables are curated in-repo: label INTERPRETED, the
impact under it measured).

Mirrors packages/trace-schema/src/counterfactuals.ts 1:1 (same dictionary,
same applicability rules) — keep both sides in sync.
"""

from __future__ import annotations

import re
from typing import TypedDict


class CounterfactualEntry(TypedDict):
    word: str
    replacement: str
    variable: str


COUNTERFACTUAL_DICTIONARY: list[CounterfactualEntry] = [
    {"word": "beginner", "replacement": "veteran", "variable": "experience"},
    {"word": "expert", "replacement": "beginner", "variable": "experience"},
    {"word": "simple", "replacement": "complex", "variable": "task complexity"},
    {"word": "easy", "replacement": "hard", "variable": "difficulty"},
    {"word": "fast", "replacement": "slow", "variable": "performance"},
    {"word": "slow", "replacement": "fast", "variable": "performance"},
    {"word": "quickly", "replacement": "carefully", "variable": "learning speed"},
    {"word": "python", "replacement": "rust", "variable": "language"},
    {"word": "rust", "replacement": "python", "variable": "language"},
    {"word": "hobby", "replacement": "job", "variable": "career goal"},
    {"word": "why", "replacement": "how", "variable": "question type"},
    {"word": "sky", "replacement": "ocean", "variable": "subject"},
    {"word": "blue", "replacement": "green", "variable": "subject"},
]

# cap per "investigate" run — each substitution is a full greedy rerun.
# Mirrored by MAX_COUNTERFACTUALS in counterfactuals.ts.
MAX_COUNTERFACTUALS = 6

# label for free-form (user-edited) counterfactuals
CUSTOM_VARIABLE = "your edit"


class ResolvedVariable(TypedDict):
    """One dictionary entry resolved against a concrete prompt."""

    variable: str
    originalWord: str
    replacementWord: str
    promptText: str


def _match_case(replacement: str, original: str) -> str:
    """Match the original's capitalization: Python→Rust, python→rust."""
    first = original[:1]
    if first.isupper() and first.lower() != first:
        return replacement[:1].upper() + replacement[1:]
    return replacement


def applicable_substitutions(prompt: str) -> list[ResolvedVariable]:
    """Every dictionary word present in the prompt (whole-word,
    case-insensitive, first occurrence), dictionary order, capped at
    MAX_COUNTERFACTUALS. One single-variable edit per entry — swapping one
    word at a time is what makes the impact attributable to a variable.

    Mirrored by applicableSubstitutions() in counterfactuals.ts.
    """
    found: list[ResolvedVariable] = []
    for entry in COUNTERFACTUAL_DICTIONARY:
        if len(found) >= MAX_COUNTERFACTUALS:
            break
        match = re.search(rf"\b{re.escape(entry['word'])}\b", prompt, re.IGNORECASE)
        if match is None:
            continue
        replacement = _match_case(entry["replacement"], match.group(0))
        found.append(
            {
                "variable": entry["variable"],
                "originalWord": match.group(0),
                "replacementWord": replacement,
                "promptText": (
                    prompt[: match.start()] + replacement + prompt[match.end() :]
                ),
            }
        )
    return found
