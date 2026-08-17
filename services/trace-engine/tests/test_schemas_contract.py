"""Contract test: the Pydantic mirror accepts every committed fixture.

These files are ALSO validated by the TypeScript suite (vitest) — both
runtimes green on the same corpus is the whole contract.
"""

from __future__ import annotations

import pytest

from app.schemas.trace import Trace


def test_every_fixture_parses(fixture_traces: list[dict]) -> None:
    for raw in fixture_traces:
        trace = Trace.model_validate(raw)
        assert trace.id.startswith("tr_")
        assert trace.events, "fixture must contain events"


def test_seq_is_gapless(fixture_traces: list[dict]) -> None:
    for raw in fixture_traces:
        seqs = [e["seq"] for e in raw["events"]]
        assert seqs == list(range(len(seqs)))


def test_extra_fields_rejected(fixture_traces: list[dict]) -> None:
    raw = dict(fixture_traces[0])
    raw["surprise"] = True
    with pytest.raises(Exception):
        Trace.model_validate(raw)


def test_roundtrip_is_stable(fixture_traces: list[dict]) -> None:
    for raw in fixture_traces:
        trace = Trace.model_validate(raw)
        again = Trace.model_validate(trace.model_dump())
        assert again == trace
