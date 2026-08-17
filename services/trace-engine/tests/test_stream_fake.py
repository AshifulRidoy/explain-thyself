"""End-to-end stream test with the FakeBackend (no torch, no DB).

Verifies the full pipeline shape: INPUT → (TOKEN, LAYER_ACTIVITY)* →
DECISION → OUTPUT → done, with gapless seq and monotone t.
"""

from __future__ import annotations

import json

import pytest

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


async def collect_fake_stream(max_tokens: int = 10) -> tuple[list, list]:
    backend = FakeBackend(MODEL_REGISTRY["fake"], seed=99)
    raw = ""
    async for frame in trace_stream(
        prompt="Why is the sky blue?",
        trace_mode="STANDARD",
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
    for i in range(1, len(types) - 2):
        if i % 2 == 1:
            assert types[i] == "TOKEN"
        else:
            assert types[i] == "LAYER_ACTIVITY"


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
    assert "TOKEN" in types
