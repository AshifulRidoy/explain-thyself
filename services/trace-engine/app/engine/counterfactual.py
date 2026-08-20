"""Counterfactual runs (spec §23 / Phase 6): WHAT WOULD CHANGE THE ANSWER?

Re-runs the answer GREEDY under one edited prompt and compares it
token-by-token against the original trace's emitted tokens. The original
trace is immutable — a comparison is a separate artifact persisted in the
`counterfactuals` table, never an event appended to the trace.

Same rerun conventions as ANSWER_STABILITY: a variant that stops or hits
the context limit early leaves its remaining positions diverged (the
answer changed, so it counts as changed).
"""

from __future__ import annotations

import asyncio
import time

from nanoid import generate as nanoid

from ..aggregation.stats import entropy_from_log_probs
from ..config import settings
from ..models.backend import ModelBackend, TopToken
from ..schemas.trace import CounterfactualResult
from .generate import _compose_output_text

_LOWER_ALPHANUM = "0123456789abcdefghijklmnopqrstuvwxyz"


def new_counterfactual_id() -> str:
    return "cf_" + nanoid(_LOWER_ALPHANUM, size=8)


class CounterfactualError(Exception):
    """Request-shaped problem (bad scope, empty trace) — maps to HTTP 400."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _original_tokens(trace: dict) -> tuple[list[int], list[float]]:
    """The original run's emitted token ids and SHIPPED 4dp entropies.

    Shipped values on purpose: the comparison must be auditable from the
    trace's own TOKEN events, exactly like MODEL_UNCERTAINTY.
    """
    token_events = [e for e in trace["events"] if e.get("type") == "TOKEN"]
    ids = [e["tokenId"] for e in token_events]
    entropies = [float(e["entropyBits"]) for e in token_events]
    return ids, entropies


async def run_counterfactual(
    trace: dict,
    *,
    prompt_text: str,
    variable: str,
    original_word: str | None,
    replacement_word: str | None,
    backend: ModelBackend,
) -> CounterfactualResult:
    """One edited prompt → one comparison against the original answer.

    Hooks and cache are OFF for the rerun (BASIC semantics): a
    counterfactual needs the answer, not the instrumentation.
    """
    orig_ids, orig_entropy = _original_tokens(trace)
    total = len(orig_ids)
    if total == 0:
        raise CounterfactualError(
            "no-tokens", "the original trace emitted no tokens — nothing to compare"
        )

    ctx = backend.encode(prompt_text)  # type: ignore[attr-defined]
    emitted: list[TopToken] = []
    entropies: list[float] = []
    diverged: list[int] = []
    agreed = 0
    ended = False
    for i, original_id in enumerate(orig_ids):
        if ended or len(ctx) >= settings.max_context:
            diverged.append(i)
            continue
        res = await asyncio.to_thread(backend.step, ctx, False, False)
        entropy = entropy_from_log_probs(res.full_log_probs)
        token_id = res.top_k[0].token_id
        if token_id == original_id:
            agreed += 1
        else:
            diverged.append(i)
        emitted.append(res.top_k[0])
        entropies.append(round(float(entropy), 4))
        ctx.append(token_id)
        if backend.is_eos(token_id):  # type: ignore[attr-defined]
            ended = True

    original_mean = sum(orig_entropy) / total
    variant_mean = sum(entropies) / len(entropies) if entropies else 0.0

    if original_word is not None and replacement_word is not None:
        edit = f"with '{original_word}'→'{replacement_word}' ({variable})"
    else:
        edit = "of your edited prompt"
    basis = (
        f"greedy rerun {edit}, compared token-by-token against the original "
        f"answer's {total} emitted tokens; impact = 1 − agreement; "
        "word-substitution sensitivity, not causal attribution"
    )

    return CounterfactualResult(
        id=new_counterfactual_id(),
        traceId=trace["id"],
        variable=variable,
        originalWord=original_word,
        replacementWord=replacement_word,
        promptText=prompt_text,
        outputText=_compose_output_text(emitted),
        tokenCount=total,
        agreedTokens=agreed,
        impact=round(1 - agreed / total, 4),
        firstDivergence=diverged[0] if diverged else None,
        entropyDelta=round(variant_mean - original_mean, 4),
        basis=basis,
        createdAt=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    )
