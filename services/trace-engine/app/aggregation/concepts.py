"""The concept dictionary — Phase 5's INTERPRETED layer (spec §20, §38).

A concept is a HUMAN-AUTHORED set of vocabulary words. Its score on a
step is exact: the probability mass the model's full next-token
distribution places on those words' token ids. The mass is MEASURED;
the claim that the word set deserves its label is the INTERPRETATION.

This file mirrors packages/trace-schema/src/concepts.ts (the authored
source) 1:1 — keep them in sync. Rules both sides enforce:

  - every word must be a single GPT-2 token in the leading-space form
    (" word") — the form that realizes the word in running text, which
    is where next-token mass lives. The bare form is used too when it
    is also a single token (about 2/3 of words; GPT-2 merges are
    space-prefixed). Model-marked tests reject words that fail this.
  - word sets are disjoint: mass attributes to exactly one label
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ConceptSpec:
    conceptId: str
    label: str
    words: tuple[str, ...]


CONCEPT_ACTIVE_MASS = 0.05
CONCEPT_EVIDENCE_LIMIT = 3

CONCEPT_DICTIONARY: tuple[ConceptSpec, ...] = (
    ConceptSpec(
        conceptId="concept_uncertainty",
        label="uncertainty / hedging",
        words=(
            "maybe", "perhaps", "might", "could", "possibly", "probably",
            "likely", "unclear", "uncertain", "depends", "sometimes",
            "generally", "usually", "often", "seems",
        ),
    ),
    ConceptSpec(
        conceptId="concept_certainty",
        label="certainty / emphasis",
        words=(
            "definitely", "certainly", "always", "must", "clearly",
            "obviously", "truly", "certain", "surely", "indeed", "simply",
            "exactly",
        ),
    ),
    ConceptSpec(
        conceptId="concept_comparison",
        label="comparison / choice",
        words=(
            "better", "worse", "best", "worst", "versus", "compared",
            "difference", "similar", "instead", "alternative", "prefer",
            "choice", "trade", "options",
        ),
    ),
    ConceptSpec(
        conceptId="concept_causality",
        label="causality / explanation",
        words=(
            "because", "since", "therefore", "thus", "causes", "caused",
            "leads", "result", "results", "hence", "due", "reason", "why",
            "explains", "effect",
        ),
    ),
    ConceptSpec(
        conceptId="concept_agency",
        label="people / agency",
        words=(
            "people", "person", "user", "human", "humans", "everyone",
            "someone", "readers", "students", "developers", "workers",
        ),
    ),
    ConceptSpec(
        conceptId="concept_quantity",
        label="quantity / degree",
        words=(
            "many", "few", "most", "several", "number", "numbers",
            "percent", "half", "twice", "majority", "minority", "lots",
            "plenty",
        ),
    ),
    ConceptSpec(
        conceptId="concept_time",
        label="time / change",
        words=(
            "today", "now", "soon", "later", "eventually", "years", "year",
            "decade", "century", "past", "future", "history", "before",
            "after", "changes", "changed",
        ),
    ),
    ConceptSpec(
        conceptId="concept_technology",
        label="technology / systems",
        words=(
            "computer", "computers", "software", "code", "data", "system",
            "systems", "model", "models", "machine", "machines", "digital",
            "program", "programs", "network", "tool", "tools", "hardware",
        ),
    ),
    ConceptSpec(
        conceptId="concept_science",
        label="science / nature",
        words=(
            "light", "energy", "climate", "weather", "air", "water",
            "earth", "physics", "science", "biology", "natural",
            "molecules", "atoms", "chemical", "sun", "sunlight", "color",
            "colors", "waves",
        ),
    ),
    ConceptSpec(
        conceptId="concept_language",
        label="language / learning",
        words=(
            "language", "languages", "word", "words", "meaning", "learn",
            "learning", "read", "reading", "write", "writing", "text",
            "sentence", "grammar", "syntax", "practice", "teaching",
        ),
    ),
)


# ---------------------------------------------------------------- scoring


@dataclass(frozen=True)
class ConceptScore:
    """One concept measured against one next-token distribution."""

    spec: ConceptSpec
    # exact probability mass on the concept's token ids — MEASURED
    mass: float
    # top tokens carrying that mass, descending — keeps the label auditable
    evidence: list[tuple[int, str, float]] = field(default_factory=list)


class ConceptScorer:
    """Scores the dictionary against full next-token log-probabilities.

    Exact by construction: mass = Σ p(token ids of the concept's words).
    No embeddings, no cosine similarity — the mass is real and the label is
    the interpretation, which is why CONCEPT events carry their evidence.
    """

    def __init__(
        self,
        word_token_ids: Callable[[str], list[int]],
        token_text: Callable[[int], str],
    ) -> None:
        self._token_text = token_text
        self._concepts: list[tuple[ConceptSpec, tuple[int, ...]]] = []
        for spec in CONCEPT_DICTIONARY:
            ids = tuple(
                sorted({i for word in spec.words for i in word_token_ids(word)})
            )
            if ids:  # a concept whose every word is multi-token scores nothing
                self._concepts.append((spec, ids))

    def score(self, log_probs: np.ndarray) -> list[ConceptScore]:
        probs = np.exp(np.asarray(log_probs, dtype=np.float64))
        scores: list[ConceptScore] = []
        for spec, ids in self._concepts:
            mass = float(probs[list(ids)].sum())
            # evidence = the tokens actually carrying the mass; a floor-level
            # tail token is not evidence of anything (> 1e-4 also survives
            # 4-dp rounding, so shipped evidence is never 0.0000)
            ranked = sorted(
                ((int(i), float(probs[i])) for i in ids if probs[i] > 1e-4),
                key=lambda t: -t[1],
            )[:CONCEPT_EVIDENCE_LIMIT]
            scores.append(
                ConceptScore(
                    spec=spec,
                    mass=mass,
                    evidence=[(i, self._token_text(i), p) for i, p in ranked],
                )
            )
        return scores
