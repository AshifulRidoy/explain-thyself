"""Concept dictionary invariants + cross-runtime sync (Phase 5).

The dictionary is authored twice — concepts.ts (canonical) and this
package's concepts.py mirror — exactly like the trace schema itself.
These tests keep the two honest: same concepts, same words, same
threshold, and no word claiming two labels.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from app.aggregation.concepts import (
    CONCEPT_ACTIVE_MASS,
    CONCEPT_DICTIONARY,
    ConceptScorer,
)
from app.schemas.trace import ConceptEvent

_TS_SOURCE = Path(__file__).resolve().parents[3] / (
    "packages/trace-schema/src/concepts.ts"
)


def _parse_ts_dictionary(source: str) -> dict[str, list[str]]:
    """Extract {conceptId: [words]} from the authored TS source.

    Deliberately dumb: the file is authored in a fixed shape, and a format
    drift failing here is a prompt to fix the parser — not a silent pass.
    """
    concepts: dict[str, list[str]] = {}
    for block in re.finditer(
        r'conceptId:\s*"([^"]+)",\s*label:\s*"[^"]*",\s*words:\s*\[(.*?)\]',
        source,
        re.DOTALL,
    ):
        concept_id = block.group(1)
        words = re.findall(r'"([a-z]+)"', block.group(2))
        concepts[concept_id] = words
    assert concepts, "concepts.ts parser found no concepts — format drifted?"
    return concepts


def test_mirror_matches_authored_ts_source():
    ts = _parse_ts_dictionary(_TS_SOURCE.read_text())
    py = {c.conceptId: list(c.words) for c in CONCEPT_DICTIONARY}
    assert set(ts) == set(py), "concept ids drifted between runtimes"
    for concept_id in py:
        assert ts[concept_id] == py[concept_id], f"words drifted: {concept_id}"


def test_word_sets_are_disjoint():
    seen: dict[str, str] = {}
    for concept in CONCEPT_DICTIONARY:
        for word in concept.words:
            owner = seen.setdefault(word, concept.conceptId)
            assert owner == concept.conceptId, (
                f"{word!r} belongs to both {owner} and {concept.conceptId}"
            )


def test_dictionary_shape():
    assert len(CONCEPT_DICTIONARY) >= 10
    for concept in CONCEPT_DICTIONARY:
        assert len(concept.words) >= 8
        assert re.fullmatch(r"concept_[a-z_]+", concept.conceptId)
    assert 0 < CONCEPT_ACTIVE_MASS < 1


def test_pydantic_accepts_evidence():
    event = ConceptEvent(
        id="evt_0001",
        seq=1,
        type="CONCEPT",
        t=10,
        level="INTERPRETED",
        conceptId="concept_uncertainty",
        label="uncertainty / hedging",
        score=0.42,
        positions=[7],
        evidence=[
            {"tokenId": 1, "text": "maybe", "probability": 0.4},
            {"tokenId": 2, "text": "perhaps", "probability": 0.02},
        ],
    )
    assert event.evidence is not None and len(event.evidence) == 2
    assert event.score == pytest.approx(0.42)


def test_scorer_mass_is_exact_sum_over_token_ids():
    """The score must be the plain probability sum on the concept's ids —
    no embedding, no similarity, nothing to tune."""
    # fake vocabulary of 100 ids; hand it a resolver where "maybe"→id 7,
    # "perhaps"→id 9, and every other dictionary word is multi-token ([]).
    resolution = {"maybe": [7], "perhaps": [9]}
    scorer = ConceptScorer(
        word_token_ids=lambda w: resolution.get(w, []),
        token_text=lambda i: {7: "maybe", 9: "perhaps"}[i],
    )

    probs = np.full(100, 1e-30)  # log-friendly zero
    probs[7] = 0.20
    probs[9] = 0.05
    probs[0] = 0.70  # the non-concept mass
    scores = scorer.score(np.log(probs))

    uncertainty = next(s for s in scores if s.spec.conceptId == "concept_uncertainty")
    assert uncertainty.mass == pytest.approx(0.25, abs=1e-9)
    assert [(i, text) for i, text, _ in uncertainty.evidence] == [(7, "maybe"), (9, "perhaps")]
    # every concept whose words resolve to nothing scores exactly 0
    assert all(s.mass == 0 for s in scores if s.spec.conceptId != "concept_uncertainty")
    # a threshold cut on these scores keeps exactly the concepts above it
    assert sum(s.mass >= CONCEPT_ACTIVE_MASS for s in scores) == 1
