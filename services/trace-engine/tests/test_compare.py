"""Cross-model comparison math (spec Phase 7, V2 scope) on the fake
backend — no database, no torch. The control rerun proves the pipeline
measures model difference, not nondeterminism; the crafted-token tests
pin the arithmetic (agreement over min length, first divergence, signed
entropy delta, length asymmetry).
"""

from __future__ import annotations

import json

import pytest

from app.engine.comparison import _split_frame, derive_comparison
from app.engine.generate import trace_stream
from app.models.fake_backend import FakeBackend
from app.models.registry import MODEL_REGISTRY

pytestmark = pytest.mark.asyncio


async def _run_fake(prompt: str, seed: int, max_tokens: int = 6) -> dict:
    """One full fake run → envelope id, token ids, shipped entropies,
    output text (collected from the frames themselves, like the engine)."""
    backend = FakeBackend(MODEL_REGISTRY["fake"], seed=seed)
    run: dict = {"id": "", "ids": [], "entropies": [], "output": ""}
    async for frame in trace_stream(
        prompt=prompt,
        trace_mode="STANDARD",
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=None,
        seed=None,
        backend=backend,
        writer=None,
    ):
        event, data = _split_frame(frame)
        if event == "trace":
            run["id"] = json.loads(data)["id"]
        elif event == "trace_event":
            payload = json.loads(data)
            if payload.get("type") == "TOKEN":
                run["ids"].append(payload["tokenId"])
                run["entropies"].append(float(payload["entropyBits"]))
            elif payload.get("type") == "OUTPUT":
                run["output"] = payload.get("text", "")
    assert run["id"] and run["ids"], "fake run produced no tokens"
    return run


def _trace_from(run: dict) -> dict:
    """The anchor-trace shape derive_comparison reads."""
    events = [{"type": "TOKEN", "tokenId": tid, "entropyBits": h}
              for tid, h in zip(run["ids"], run["entropies"])]
    events.append({"type": "OUTPUT", "text": run["output"]})
    return {
        "id": run["id"],
        "model": {"name": "fake"},
        "input": {"text": "Why is the sky blue?"},
        "traceMode": "STANDARD",
        "sampling": {"maxTokens": len(run["ids"]), "temperature": 0.0,
                     "topK": None, "seed": None},
        "output": {"text": run["output"]},
        "events": events,
    }


async def test_control_rerun_agrees_exactly():
    """The same fake model twice (fresh seeded backends): agreement 1.0,
    no divergence, zero entropy shift — divergence in a real comparison
    is then attributable to the MODEL, not to noise."""
    run_a = await _run_fake("Why is the sky blue?", seed=7)
    run_b = await _run_fake("Why is the sky blue?", seed=7)
    result = derive_comparison(
        _trace_from(run_a),
        model_b="fake",
        trace_b_id=run_b["id"],
        ids_b=run_b["ids"],
        entropy_b=run_b["entropies"],
        output_b=run_b["output"],
    )
    assert result.agreement == 1.0
    assert result.firstDivergence is None
    assert result.entropyDelta == 0.0
    assert result.comparedLength == result.tokenCountA == result.tokenCountB
    assert "not internal similarity" in result.basis


async def test_divergence_math_is_exact():
    """Crafted ids: B diverges at position 2 of 5 → agreement 3/5, and a
    signed entropy shift that flows through the rounding unchanged."""
    trace = _trace_from({
        "id": "tr_anchor00", "ids": [10, 11, 12, 13, 14],
        "entropies": [2.0, 2.0, 2.0, 2.0, 2.0], "output": "aaaaa",
    })
    result = derive_comparison(
        trace,
        model_b="other",
        trace_b_id="tr_other000",
        ids_b=[10, 11, 99, 13, 14],
        entropy_b=[3.0, 3.0, 3.0, 3.0, 3.0],
        output_b="bbbbb",
    )
    assert result.comparedLength == 5
    assert result.agreedTokens == 4
    assert result.agreement == 0.8
    assert result.firstDivergence == 2
    assert result.meanEntropyA == 2.0 and result.meanEntropyB == 3.0
    assert result.entropyDelta == 1.0
    assert result.tokenCountA == result.tokenCountB == 5


async def test_length_asymmetry_is_not_divergence():
    """B stops earlier but agrees on its whole prefix: compared length is
    min(lenA, lenB), agreement is 1.0, and BOTH counts ship — a length
    difference is visible next to the score, not hidden inside it."""
    trace = _trace_from({
        "id": "tr_anchor00", "ids": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "entropies": [1.0] * 10, "output": "a" * 10,
    })
    result = derive_comparison(
        trace,
        model_b="other",
        trace_b_id="tr_other000",
        ids_b=[10, 11, 12, 13],
        entropy_b=[1.0, 1.0, 1.0, 1.0],
        output_b="aaaa",
    )
    assert result.comparedLength == 4
    assert result.agreement == 1.0
    assert result.firstDivergence is None
    assert result.tokenCountA == 10 and result.tokenCountB == 4


async def test_basis_names_both_models_and_the_limit():
    trace = _trace_from({
        "id": "tr_anchor00", "ids": [10, 11], "entropies": [1.0, 1.0],
        "output": "ab",
    })
    result = derive_comparison(
        trace, model_b="distilgpt2", trace_b_id="tr_other000",
        ids_b=[10, 50], entropy_b=[2.0, 2.0], output_b="zz",
    )
    assert "distilgpt2" in result.basis and "fake" in result.basis
    assert "first 2 positions" in result.basis
