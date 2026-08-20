"""Cross-model comparison (spec Phase 7, V2 scope): the same prompt
through two REGISTERED models.

Model B answers the prompt as a completely normal recorded trace — same
pipeline, its own events, its own replay — by iterating `trace_stream`
and reading the frames it yields (the frames ARE the contract, so no
generation logic is duplicated here). The comparison itself is then
DERIVED from both traces' TOKEN events and persisted as a separate
artifact in the `comparisons` table: neither trace is ever mutated.

Honesty rules this module enforces:

* agreement is only computed when the two models share a tokenizer —
  the ROUTE rejects mismatched tokenizers before anything runs here;
  ids across different vocabularies would be a number about nothing
* compared positions = min(lenA, lenB): each answer keeps its own
  length; a model is not penalized for stopping earlier, and the length
  difference ships next to the agreement so nothing is hidden
* entropies come from the SHIPPED entropyBits of both traces, so the
  whole artifact is recomputable from the trace JSON
* model B always runs greedy — agreement is a stability-style measure
  and only means anything under determinism
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator

import asyncpg
from nanoid import generate as nanoid

from ..models.backend import ModelBackend
from ..schemas.trace import ComparisonResult
from ..storage.comparison_store import save_comparison
from ..storage.trace_writer import TraceWriter
from .generate import trace_stream
from .sse import format_sse

_LOWER_ALPHANUM = "0123456789abcdefghijklmnopqrstuvwxyz"


def new_comparison_id() -> str:
    return "cmp_" + nanoid(_LOWER_ALPHANUM, size=8)


def _recorded_tokens(trace: dict) -> tuple[list[int], list[float]]:
    """The anchor trace's emitted token ids and SHIPPED 4dp entropies —
    auditable from the trace's own TOKEN events, nothing recomputed."""
    token_events = [e for e in trace["events"] if e.get("type") == "TOKEN"]
    ids = [e["tokenId"] for e in token_events]
    entropies = [float(e["entropyBits"]) for e in token_events]
    return ids, entropies


def _split_frame(frame: str) -> tuple[str | None, str]:
    """One SSE frame → (event name, joined data). Hand-rolled inverse of
    format_sse; json.dumps never emits raw newlines, so data is one line."""
    event: str | None = None
    data_lines: list[str] = []
    for line in frame.splitlines():
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: ") :])
    return event, "\n".join(data_lines)


def derive_comparison(
    trace: dict,
    *,
    model_b: str,
    trace_b_id: str,
    ids_b: list[int],
    entropy_b: list[float],
    output_b: str,
) -> ComparisonResult:
    """The DERIVED half of the artifact: agreement, divergence, entropy
    contrast — everything recomputable from the two traces' shipped
    TOKEN events. Pure, so the math is testable without a database."""
    ids_a, entropy_a = _recorded_tokens(trace)
    model_a = trace["model"]["name"]

    compared = min(len(ids_a), len(ids_b))
    agreed = sum(1 for a, b in zip(ids_a, ids_b) if a == b)
    divergence = next(
        (i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b), None
    )
    mean_a = sum(entropy_a) / len(entropy_a)
    mean_b = sum(entropy_b) / len(entropy_b)

    basis = (
        f"same prompt rerun greedy through {model_b} ({len(ids_b)} tokens) and "
        f"compared position-by-position against {model_a}'s recorded answer "
        f"({len(ids_a)} tokens) over the first {compared} positions; agreement "
        "is shared token overlap under a common tokenizer — surface behavior, "
        "not internal similarity; the models' internals are compared only "
        "through their own traces"
    )

    return ComparisonResult(
        id=new_comparison_id(),
        traceIdA=trace["id"],
        traceIdB=trace_b_id,
        modelA=model_a,
        modelB=model_b,
        prompt=trace["input"]["text"],
        tokenCountA=len(ids_a),
        tokenCountB=len(ids_b),
        comparedLength=compared,
        agreedTokens=agreed,
        agreement=round(agreed / compared, 4),
        firstDivergence=divergence,
        # envelope ships "output": null until completion; rows loaded from
        # the DB (status complete) carry the text
        outputTextA=(trace.get("output") or {}).get("text", ""),
        outputTextB=output_b,
        meanEntropyA=round(mean_a, 4),
        meanEntropyB=round(mean_b, 4),
        entropyDelta=round(mean_b - mean_a, 4),
        basis=basis,
        createdAt=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
    )


async def comparison_stream(
    trace: dict,
    *,
    backend: ModelBackend,
    writer: TraceWriter,
    pool: asyncpg.Pool,
) -> AsyncGenerator[str, None]:
    """Yield the comparison SSE stream.

    Frames: `event: progress` (model B's token count as its answer grows —
    transport chatter, not a contract artifact), then one
    `event: comparison` (the ComparisonResult JSON), then exactly one
    terminal `event: done|error`. The artifact is persisted before the
    comparison frame is yielded, so a client that saw it can always
    restore it.
    """
    ids_a, _ = _recorded_tokens(trace)  # route guarantees ≥1; keep the shape local
    model_b = backend.spec.key  # type: ignore[attr-defined]
    prompt = trace["input"]["text"]

    ids_b: list[int] = []
    entropy_b: list[float] = []
    output_b = ""
    trace_b_id = ""
    failure: dict | None = None

    try:
        if not ids_a:  # unreachable through the route (_load_complete_trace
            yield format_sse(  # guards it); returning here still stops the
                "error",  # writer via the finally below
                json.dumps(
                    {
                        "code": "trace-a-empty",
                        "message": "the anchor trace emitted no tokens — nothing to compare",
                    }
                ),
            )
            return
        async for frame in trace_stream(
            prompt=prompt,
            trace_mode=trace["traceMode"],
            max_tokens=trace["sampling"]["maxTokens"],
            temperature=0.0,
            top_k=None,
            seed=None,
            backend=backend,
            writer=writer,
        ):
            event, data = _split_frame(frame)
            if event == "trace":
                trace_b_id = json.loads(data)["id"]
            elif event == "error":
                failure = json.loads(data)
            elif event == "trace_event":
                payload = json.loads(data)
                kind = payload.get("type")
                if kind == "TOKEN":
                    ids_b.append(payload["tokenId"])
                    entropy_b.append(float(payload["entropyBits"]))
                    yield format_sse(
                        "progress",
                        json.dumps({"model": model_b, "tokenCount": len(ids_b)}),
                    )
                elif kind == "OUTPUT":
                    output_b = payload.get("text", "")
    finally:
        # shielded like POST /trace: a client disconnect cancelling this
        # generator must not orphan model B's trace row
        stop = asyncio.create_task(writer.stop())
        try:
            await asyncio.shield(stop)
        except asyncio.CancelledError:
            pass

    if failure is not None:
        yield format_sse("error", json.dumps(failure))
        return
    if not ids_b:
        yield format_sse(
            "error",
            json.dumps(
                {
                    "code": "model-b-empty",
                    "message": f"{model_b} emitted no tokens for this prompt",
                }
            ),
        )
        return

    result = derive_comparison(
        trace,
        model_b=model_b,
        trace_b_id=trace_b_id,
        ids_b=ids_b,
        entropy_b=entropy_b,
        output_b=output_b,
    )

    try:
        await save_comparison(pool, result)
    except Exception as exc:  # noqa: BLE001 — the stream must terminate cleanly
        yield format_sse(
            "error",
            json.dumps(
                {"code": "comparison_not_saved", "message": str(exc)}
            ),
        )
        return

    yield format_sse("comparison", result.model_dump_json())
    yield format_sse(
        "done",
        json.dumps(
            {
                "traceId": trace["id"],
                "comparisonId": result.id,
                "traceIdB": trace_b_id,
            }
        ),
    )
