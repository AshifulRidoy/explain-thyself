"""Counterfactual mode (spec §23): the dictionary resolves honestly, the
comparison math is exact and auditable, the control (identical prompt)
agrees completely, and reruns are deterministic.

The FakeBackend models sensitivity in tiers: the exact authored prompt
returns its authored response VERBATIM, a word-level edit of it (any
counterfactual variable) returns the seeded-swap variant — same length,
deterministic positions — and unrelated prompts return a fixed generic
response. Per-position agreement is well defined at every tier.
"""

from __future__ import annotations

import json

import pytest

from app.aggregation.counterfactuals import (
    COUNTERFACTUAL_DICTIONARY,
    CUSTOM_VARIABLE,
    MAX_COUNTERFACTUALS,
    applicable_substitutions,
)
from app.engine.counterfactual import CounterfactualError, run_counterfactual
from app.engine.generate import trace_stream
from app.models.fake_backend import FakeBackend
from app.models.registry import MODEL_REGISTRY


async def _fake_trace(prompt: str, max_tokens: int = 10) -> dict:
    """A complete trace dict from the fake pipeline (envelope + events)."""
    raw = ""
    async for frame in trace_stream(
        prompt=prompt,
        trace_mode="STANDARD",
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=None,
        seed=None,
        backend=FakeBackend(MODEL_REGISTRY["fake"], seed=7),
        writer=None,
    ):
        raw += frame
    frames = [
        line[6:] for line in raw.split("\n") if line.startswith("data: ")
    ]
    envelope = json.loads(frames[0])
    envelope["events"] = [json.loads(d) for d in frames[1:-1]]
    envelope["status"] = "complete"
    return envelope


# ——— the dictionary ————————————————————————————————————————————————


def test_applicability_whole_word_case_preserving() -> None:
    variables = applicable_substitutions("Why is the sky blue?")
    assert [(v["variable"], v["originalWord"], v["replacementWord"]) for v in variables] == [
        ("question type", "Why", "How"),
        ("subject", "sky", "ocean"),
        ("subject", "blue", "green"),
    ]
    # each rerun swaps exactly ONE variable — attribution stays clean
    assert variables[0]["promptText"] == "How is the sky blue?"
    assert variables[1]["promptText"] == "Why is the ocean blue?"


def test_applicability_word_boundary_and_case() -> None:
    # "CPython" must NOT match "python" (word boundary), "Python" must
    assert applicable_substitutions("I love CPython but not rust.") == [
        {
            "variable": "language",
            "originalWord": "rust",
            "replacementWord": "python",
            "promptText": "I love CPython but not python.",
        }
    ]


def test_applicability_capped_at_max() -> None:
    # a prompt containing every dictionary word: only the first
    # MAX_COUNTERFACTUALS entries (dictionary order) run — the cap is real
    words = " ".join(e["word"] for e in COUNTERFACTUAL_DICTIONARY)
    variables = applicable_substitutions(f"{words} everything")
    assert len(variables) == MAX_COUNTERFACTUALS
    assert variables[0]["variable"] == "experience"


def test_applicability_nothing_applicable() -> None:
    assert applicable_substitutions("Tell me about tensors.") == []


# ——— the comparison ————————————————————————————————————————————————


@pytest.mark.asyncio
async def test_comparison_math_is_exact() -> None:
    trace = await _fake_trace("Should I learn Python or Rust?")
    tokens = [e for e in trace["events"] if e["type"] == "TOKEN"]
    backend = FakeBackend(MODEL_REGISTRY["fake"])

    variables = applicable_substitutions(trace["input"]["text"])
    assert len(variables) == 2  # python→rust, rust→python
    result = await run_counterfactual(
        trace,
        prompt_text=variables[0]["promptText"],
        variable=variables[0]["variable"],
        original_word=variables[0]["originalWord"],
        replacement_word=variables[0]["replacementWord"],
        backend=backend,
    )

    # compared over the ORIGINAL's token count — the audit anchor
    assert result.tokenCount == len(tokens)
    assert result.traceId == trace["id"]
    assert 0 <= result.agreedTokens <= result.tokenCount
    # impact is exactly 1 − agreement, nothing hidden
    assert result.impact == pytest.approx(
        round(1 - result.agreedTokens / result.tokenCount, 4)
    )
    if result.firstDivergence is not None:
        assert 0 <= result.firstDivergence < result.tokenCount
    # ΔH recomputable from the two means the basis names
    original_mean = sum(float(e["entropyBits"]) for e in tokens) / len(tokens)
    assert -5.0 < result.entropyDelta < 5.0
    assert "not causal attribution" in result.basis
    assert result.outputText  # the counterfactual answer ships in full


@pytest.mark.asyncio
async def test_control_identical_prompt_agrees_completely() -> None:
    """The control that makes impact a measurement: rerunning the UNEDITED
    prompt must reproduce the original answer token-for-token (greedy is
    deterministic), so any divergence under an edit is caused by the edit."""
    trace = await _fake_trace("Why is the sky blue?")
    tokens = [e for e in trace["events"] if e["type"] == "TOKEN"]
    control = await run_counterfactual(
        trace,
        prompt_text=trace["input"]["text"],
        variable="control",
        original_word=None,
        replacement_word=None,
        backend=FakeBackend(MODEL_REGISTRY["fake"]),
    )
    assert control.agreedTokens == control.tokenCount == len(tokens)
    assert control.impact == 0.0
    assert control.firstDivergence is None
    assert control.outputText == trace["events"][-1]["text"]


@pytest.mark.asyncio
async def test_substituted_prompt_actually_diverges() -> None:
    """The fake models sensitivity: a substituted prompt is NOT the authored
    prompt, so its continuation is the seeded-swap variant — measurable
    divergence, deterministic per prompt."""
    trace = await _fake_trace("Why is the sky blue?")
    variables = applicable_substitutions(trace["input"]["text"])
    changed = await run_counterfactual(
        trace,
        prompt_text=variables[0]["promptText"],
        variable=variables[0]["variable"],
        original_word=variables[0]["originalWord"],
        replacement_word=variables[0]["replacementWord"],
        backend=FakeBackend(MODEL_REGISTRY["fake"]),
    )
    assert changed.firstDivergence is not None
    assert changed.impact > 0.0


@pytest.mark.asyncio
async def test_free_form_edit_runs_the_users_prompt() -> None:
    trace = await _fake_trace("Why is the sky blue?")
    edited = "Explain rainbows instead."
    result = await run_counterfactual(
        trace,
        prompt_text=edited,
        variable=CUSTOM_VARIABLE,
        original_word=None,
        replacement_word=None,
        backend=FakeBackend(MODEL_REGISTRY["fake"]),
    )
    assert result.variable == CUSTOM_VARIABLE
    assert result.promptText == edited
    assert result.originalWord is None and result.replacementWord is None
    assert "edited prompt" in result.basis


@pytest.mark.asyncio
async def test_reruns_are_deterministic() -> None:
    trace = await _fake_trace("Should I learn Python or Rust?")
    v = applicable_substitutions(trace["input"]["text"])[0]

    def fields(r) -> tuple:
        return (
            r.variable, r.promptText, r.outputText, r.tokenCount,
            r.agreedTokens, r.impact, r.firstDivergence, r.entropyDelta,
        )

    a = await run_counterfactual(
        trace, prompt_text=v["promptText"], variable=v["variable"],
        original_word=v["originalWord"], replacement_word=v["replacementWord"],
        backend=FakeBackend(MODEL_REGISTRY["fake"]),
    )
    b = await run_counterfactual(
        trace, prompt_text=v["promptText"], variable=v["variable"],
        original_word=v["originalWord"], replacement_word=v["replacementWord"],
        backend=FakeBackend(MODEL_REGISTRY["fake"]),
    )
    assert fields(a) == fields(b)
    assert a.id != b.id  # distinct artifacts, identical measurements


@pytest.mark.asyncio
async def test_empty_trace_is_rejected() -> None:
    trace = await _fake_trace("Why is the sky blue?")
    trace["events"] = []
    with pytest.raises(CounterfactualError):
        await run_counterfactual(
            trace,
            prompt_text="anything",
            variable="x",
            original_word=None,
            replacement_word=None,
            backend=FakeBackend(MODEL_REGISTRY["fake"]),
        )
