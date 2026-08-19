"""End-to-end stream test with the FakeBackend (no torch, no DB).

Verifies the full pipeline shape: INPUT → (TOKEN, LAYER_ACTIVITY, CONCEPT×k)*
→ DECISION → OUTPUT → done, with gapless seq and monotone t. RESEARCH adds
attention per step and the uncertainty layer after OUTPUT (spec §22).
"""

from __future__ import annotations

import json
import math

import pytest

from app.aggregation.concepts import CONCEPT_ACTIVE_MASS, CONCEPT_DICTIONARY
from app.aggregation.stability import UNCERTAINTY_KINDS
from app.engine.generate import trace_stream
from app.models.fake_backend import FakeBackend
from app.models.registry import MODEL_REGISTRY
from app.schemas.trace import Trace


def parse_frames(raw: str) -> list[tuple[str | None, str]]:
    frames = []
    for block in raw.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        event = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        frames.append((event, "\n".join(data_lines)))
    return frames


async def collect_fake_stream(
    max_tokens: int = 10, trace_mode: str = "STANDARD"
) -> tuple[list, list]:
    backend = FakeBackend(MODEL_REGISTRY["fake"], seed=99)
    raw = ""
    async for frame in trace_stream(
        prompt="Why is the sky blue?",
        trace_mode=trace_mode,
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=None,
        seed=None,
        backend=backend,
        writer=None,
    ):
        raw += frame
    frames = parse_frames(raw)
    events = [json.loads(d) for e, d in frames if e == "trace_event"]
    envelope = json.loads(next(d for e, d in frames if e == "trace"))
    return frames, [envelope, *events]


@pytest.mark.asyncio
async def test_frame_order_and_termination() -> None:
    frames, messages = await collect_fake_stream()
    names = [e for e, _ in frames]
    assert names[0] == "trace"
    assert names[-1] == "done"
    assert names.count("done") == 1
    types = [m["type"] for m in messages[1:]]
    assert types[0] == "INPUT"
    assert types[-2] == "DECISION"
    assert types[-1] == "OUTPUT"
    # each step's block: TOKEN, then its derived/interpreted signals —
    # LAYER_ACTIVITY once, then CONCEPT×k (k ≥ 0), nothing else
    blocks: list[list[str]] = [[]]
    for event_type in types[1:-2]:
        if event_type == "TOKEN":
            blocks.append([])
        else:
            blocks[-1].append(event_type)
    assert len(blocks) == 11  # leading empty + 10 steps
    for block in blocks[1:]:
        assert block[0] == "LAYER_ACTIVITY"
        assert set(block[1:]) <= {"CONCEPT"}


@pytest.mark.asyncio
async def test_seq_gapless_and_t_monotone() -> None:
    _, messages = await collect_fake_stream()
    seqs = [m["seq"] for m in messages[1:]]
    assert seqs == list(range(len(seqs)))
    ts = [m["t"] for m in messages[1:]]
    assert ts == sorted(ts)


@pytest.mark.asyncio
async def test_all_events_validate_against_contract() -> None:
    frames, messages = await collect_fake_stream()
    envelope = dict(messages[0])
    envelope["events"] = messages[1:]
    trace = Trace.model_validate(envelope)  # raises on any drift
    assert trace.traceMode == "STANDARD"
    assert trace.sampling.temperature == 0


@pytest.mark.asyncio
async def test_deterministic_same_seed_same_output() -> None:
    _, a = await collect_fake_stream()
    _, b = await collect_fake_stream()
    assert [m.get("text") for m in a] == [m.get("text") for m in b]
    assert [m.get("probability") for m in a] == [m.get("probability") for m in b]


@pytest.mark.asyncio
async def test_http_endpoint_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from app.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "model", "fake")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/trace",
            json={"prompt": "Why is the sky blue?", "maxTokens": 5, "persist": False},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        frames = parse_frames(resp.text)
        names = [e for e, _ in frames]
        assert names[0] == "trace"
        assert names[-1] == "done"
        types = [json.loads(d)["type"] for e, d in frames if e == "trace_event"]
        assert "TOKEN" in types and "LAYER_ACTIVITY" in types


@pytest.mark.asyncio
async def test_basic_mode_emits_no_layer_activity() -> None:
    backend = FakeBackend(MODEL_REGISTRY["fake"], seed=5)
    raw = ""
    async for frame in trace_stream(
        prompt="Hi",
        trace_mode="BASIC",
        max_tokens=3,
        temperature=0.0,
        top_k=None,
        seed=None,
        backend=backend,
        writer=None,
    ):
        raw += frame
    types = [json.loads(d)["type"] for e, d in parse_frames(raw) if e == "trace_event"]
    assert "LAYER_ACTIVITY" not in types
    assert "CONCEPT" not in types  # concepts are a non-BASIC signal
    assert "UNCERTAINTY" not in types  # uncertainty rides RESEARCH only
    assert "TOKEN" in types


@pytest.mark.asyncio
async def test_standard_mode_emits_concepts() -> None:
    """STANDARD: steps carry INTERPRETED concept events — exact mass above
    threshold, evidence tokens drawn from the concept's own word set, score
    order descending within a step, fully deterministic."""
    _, messages = await collect_fake_stream()
    concepts = [m for m in messages[1:] if m["type"] == "CONCEPT"]
    assert concepts, "fake STANDARD traces are concept-bearing by design"

    dictionary = {c.conceptId: c for c in CONCEPT_DICTIONARY}
    prompt_len = next(m["tokenCount"] for m in messages[1:] if m["type"] == "INPUT")
    by_position: dict[int, list[dict]] = {}
    for event in concepts:
        assert event["level"] == "INTERPRETED"
        assert event["score"] >= CONCEPT_ACTIVE_MASS
        spec = dictionary[event["conceptId"]]
        assert event["label"] == spec.label
        (position,) = event["positions"]  # exactly one position per event
        by_position.setdefault(position, []).append(event)
        assert event["evidence"], "every concept ships its evidence"
        for evidence in event["evidence"]:
            assert evidence["text"] in spec.words
            assert 0 < evidence["probability"] <= event["score"]

    # within one step, events arrive sorted by mass (desc)
    for events in by_position.values():
        scores = [e["score"] for e in events]
        assert scores == sorted(scores, reverse=True)
    assert len(by_position) >= 3  # across 10 steps, several light up

    _, again = await collect_fake_stream()
    assert [m for m in again[1:] if m["type"] == "CONCEPT"] == concepts


@pytest.mark.asyncio
async def test_standard_mode_has_no_uncertainty_layer() -> None:
    """STANDARD keeps the dial honest: no attention, no stability reruns —
    the uncertainty layer is a RESEARCH signal (it re-runs the answer)."""
    _, messages = await collect_fake_stream(trace_mode="STANDARD")
    types = [m["type"] for m in messages[1:]]
    assert "ATTENTION" not in types
    assert "UNCERTAINTY" not in types


@pytest.mark.asyncio
async def test_research_mode_emits_uncertainty_layer() -> None:
    """RESEARCH: after OUTPUT, the four separated quantities (spec §22) —
    model uncertainty recomputable from the trace's own TOKEN events,
    answer stability with per-perturbation evidence, and honest nulls for
    the quantities this instrument cannot measure."""
    _, messages = await collect_fake_stream(trace_mode="RESEARCH")
    events = messages[1:]
    types = [e["type"] for e in events]

    uncertainty = [e for e in events if e["type"] == "UNCERTAINTY"]
    assert [e["kind"] for e in uncertainty] == UNCERTAINTY_KINDS
    # post-hoc: they land between OUTPUT and the terminal done frame
    assert types[-6:] == ["DECISION", "OUTPUT"] + ["UNCERTAINTY"] * 4

    model = uncertainty[0]
    tokens = [e for e in events if e["type"] == "TOKEN"]
    mean_norm = sum(t["entropyBits"] for t in tokens) / len(tokens) / math.log2(50_000)
    assert model["level"] == "DERIVED"
    assert model["value"] == pytest.approx(mean_norm, abs=1e-4)
    assert model["window"] == {"fromStep": 0, "toStep": len(tokens) - 1}

    for skipped in uncertainty[1:3]:  # EVIDENCE_QUALITY, INPUT_AMBIGUITY
        assert skipped["value"] is None
        assert skipped["level"] is None
        assert skipped["basis"].startswith("not measured")

    stability = uncertainty[3]
    assert stability["level"] == "MEASURED"
    assert 0 <= stability["value"] <= 1
    variants = stability["variants"]
    assert len(variants) == 2  # both perturbations apply to this prompt
    assert {v["perturbation"] for v in variants} == {
        "strip_final_punct",
        "lowercase_first",
    }
    total = len(variants) * len(tokens)
    agreed = sum(v["agreedTokens"] for v in variants)
    assert stability["value"] == pytest.approx(round(agreed / total, 4), abs=1e-9)
    for variant in variants:
        assert variant["text"] != "Why is the sky blue?"
        assert variant["totalTokens"] == len(tokens)
        assert variant["agreedTokens"] == variant["totalTokens"] - len(variant["divergedPositions"])
        assert variant["divergedPositions"] == sorted(set(variant["divergedPositions"]))
        assert 0 < variant["agreedTokens"] < variant["totalTokens"]  # fake diverges, but not everywhere

    _, again = await collect_fake_stream(trace_mode="RESEARCH")
    assert [e for e in again[1:] if e["type"] == "UNCERTAINTY"] == uncertainty


@pytest.mark.asyncio
async def test_research_mode_emits_attention_per_layer() -> None:
    """RESEARCH: every step emits TOKEN → LAYER_ACTIVITY → 12× ATTENTION
    (layers ascending) → CONCEPT×k; rows carry BOS at position -1 and sum
    to ~1."""
    backend = FakeBackend(MODEL_REGISTRY["fake"], seed=99)
    raw = ""
    async for frame in trace_stream(
        prompt="Why is the sky blue?",
        trace_mode="RESEARCH",
        max_tokens=3,
        temperature=0.0,
        top_k=None,
        seed=None,
        backend=backend,
        writer=None,
    ):
        raw += frame
    events = [json.loads(d) for e, d in parse_frames(raw) if e == "trace_event"]
    types = [e["type"] for e in events]

    assert types.count("ATTENTION") == 3 * 12
    # per-step block shape: TOKEN, LAYER_ACTIVITY, then L01..L12, then concepts
    first_block = types[1 : 1 + 14]
    assert first_block[0] == "TOKEN" and first_block[1] == "LAYER_ACTIVITY"
    assert [t for t in first_block[2:]] == ["ATTENTION"] * 12
    if "CONCEPT" in types:  # concepts close the step, after every layer
        rest = types[1 + 14 : types.index("TOKEN", 2)]
        assert set(rest) <= {"CONCEPT"}

    attn = [e for e in events if e["type"] == "ATTENTION"]
    assert [a["layer"] for a in attn[:12]] == list(range(1, 13))
    first = attn[0]
    assert first["level"] == "DERIVED"
    assert first["position"] == 6  # prompt_len (5) + step 0... BOS-less count
    # BOS entry: position -1, first in the row, real mass
    assert first["aggregated"][0] == {
        "position": -1,
        "text": "<bos>",
        "weight": pytest.approx(first["aggregated"][0]["weight"]),
    }
    assert first["aggregated"][0]["weight"] > 0.05
    total = sum(entry["weight"] for entry in first["aggregated"])
    assert total == pytest.approx(1.0, abs=5e-3)  # 4dp rounding tolerance
    assert len(first["headEntropyBits"]) == 12
    # seq stays gapless through the 14-event blocks
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(len(seqs)))
