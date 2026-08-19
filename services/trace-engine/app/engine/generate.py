"""The generation loop — ONE pipeline for BASIC (Phase 2) and STANDARD
(Phase 3) trace modes. Produces small validated JSON events; never tensors.

Contract: SSE frames as strings. First `event: trace` (envelope, events []),
then `event: trace_event` frames, then exactly one terminal `event: done` or
`event: error`.

The torch forward pass runs in a worker thread (asyncio.to_thread) so the
event loop — and SSE heartbeats — never block on compute.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

import numpy as np
from nanoid import generate as nanoid

from ..aggregation.concepts import CONCEPT_ACTIVE_MASS, ConceptScorer
from ..aggregation.stats import (
    RunningNormalizer,
    entropy_from_log_probs,
    sample_greedy,
    sample_with_temperature,
)
from ..config import settings
from ..models.backend import ModelBackend, TopToken
from ..schemas.trace import (
    AttentionEvent,
    ConceptEvent,
    DecisionEvent,
    InputEvent,
    InputToken,
    LayerActivityEvent,
    LayerStat,
    OutputEvent,
    TokenEvent,
    Trace,
    TraceEvent,
    TraceOutput,
)
from ..storage.trace_writer import TraceWriter
from .sse import format_sse

_LOWER_ALPHANUM = "0123456789abcdefghijklmnopqrstuvwxyz"


def new_trace_id() -> str:
    return "tr_" + nanoid(_LOWER_ALPHANUM, size=8)


def _compose_output_text(tokens: list[TopToken]) -> str:
    parts: list[str] = []
    for tok in tokens:
        prefix = ""
        if tok.leading_space and not tok.text.startswith((" ", "\n")):
            prefix = " "
        parts.append(prefix + tok.text)
    return "".join(parts)


class GenerationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


async def trace_stream(
    prompt: str,
    trace_mode: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int | None,
    backend: ModelBackend,
    writer: TraceWriter | None,
    trace_id: str | None = None,
) -> AsyncGenerator[str, None]:
    spec = backend.spec  # type: ignore[attr-defined]
    trace_id = trace_id or new_trace_id()
    started = time.perf_counter()

    rng = np.random.default_rng(seed) if (temperature or 0) > 0 and seed is not None else None

    prompt_toks = backend.prompt_tokens(prompt)  # type: ignore[attr-defined]
    prompt_len = len(prompt_toks)

    display_id: int
    if writer is not None:
        display_id = await writer.open_trace(
            trace_id=trace_id,
            model_name=spec.key,
            model_revision="",
            device=getattr(backend, "device", ""),
            trace_mode=trace_mode,
            input_text=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        display_id = int(time.time() * 1000) % 999_999 + 1

    envelope = Trace(
        id=trace_id,
        displayId=display_id,
        model={
            "name": spec.key,
            "revision": "",
            "device": getattr(backend, "device", ""),
            "layerCount": spec.layer_count,
            "paramCount": spec.param_count,
        },
        input={"text": prompt},
        traceMode=trace_mode,  # type: ignore[arg-type]
        sampling={
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topK": top_k,
            "seed": seed,
        },
        status="streaming",
        createdAt=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        events=[],
    )
    yield format_sse("trace", envelope.model_dump_json())
    if writer is not None:
        # persist what streamed so replay needs no re-derivation
        await writer.save_envelope(trace_id, json.loads(envelope.model_dump_json()))

    seq = 0
    t = 3

    def next_id() -> str:
        nonlocal seq
        ident = f"evt_{seq:04d}"
        seq += 1
        return ident

    async def emit(event: TraceEvent) -> str:
        nonlocal t
        if event.t > t:
            t = event.t
        data = event.model_dump_json()
        if writer is not None:
            await writer.enqueue_event(trace_id, json.loads(data))
        return format_sse("trace_event", data)

    try:
        input_event = InputEvent(
            id=next_id(),
            seq=seq - 1,
            type="INPUT",
            t=t,
            level="MEASURED",
            text=prompt,
            tokenCount=prompt_len,
            tokens=[
                InputToken(position=i, tokenId=tid, text=text)
                for i, (tid, text) in enumerate(prompt_toks)
            ],
        )
        yield await emit(input_event)

        ctx = backend.encode(prompt)
        collect_layers = trace_mode != "BASIC" and spec.emits_resid_activity
        collect_attention = trace_mode == "RESEARCH"
        # concepts ride the distribution the step already produced — no extra
        # forward pass, just an exact sum over the full log-probs
        scorer = (
            ConceptScorer(backend.word_token_ids, backend.token_text)  # type: ignore[attr-defined]
            if trace_mode != "BASIC"
            else None
        )
        normalizer = RunningNormalizer(spec.layer_count) if collect_layers else None
        emitted: list[TopToken] = []
        # position-indexed texts for attention rows; index 0 of ctx is the
        # BOS encode() prepends — surfaced as position -1, never dropped
        ctx_texts: list[str] = ["<bos>"] + [text for _, text in prompt_toks]
        finish_reason = "max_tokens"

        for step in range(max_tokens):
            if len(ctx) >= settings.max_context:
                finish_reason = "context_limit"
                break

            res = await asyncio.to_thread(
                backend.step, ctx, collect_layers, collect_attention
            )
            entropy = entropy_from_log_probs(res.full_log_probs)

            if (temperature or 0) > 0:
                token_id = sample_with_temperature(res.full_log_probs, temperature, top_k, rng)
                tok = next((k for k in res.top_k if k.token_id == token_id), None)
                if tok is None:  # sampled outside top-k: decode it directly
                    tok = backend.decode_token(token_id)  # type: ignore[attr-defined]
                rank = tok.rank
            else:
                tok = res.top_k[0]
                token_id = tok.token_id
                rank = 0

            latency_ms = round(res.latency_ms, 1)
            t += int(round(res.latency_ms)) + 15

            token_event = TokenEvent(
                id=next_id(),
                seq=seq - 1,
                type="TOKEN",
                t=t,
                level="MEASURED",
                position=prompt_len + step,
                step=step,
                tokenId=token_id,
                text=tok.text,
                rawText=tok.raw_text,
                leadingSpace=tok.leading_space,
                probability=round(float(tok.probability), 4),
                rank=rank,
                entropyBits=round(float(entropy), 4),
                topK=[
                    {
                        "tokenId": k.token_id,
                        "text": k.text,
                        "rawText": k.raw_text,
                        "leadingSpace": k.leading_space,
                        "probability": round(float(k.probability), 4),
                        "rank": k.rank,
                    }
                    for k in res.top_k
                ],
                latencyMs=latency_ms,
            )
            yield await emit(token_event)
            emitted.append(tok)

            if res.layer_stats and normalizer is not None:
                ratios = normalizer.update(res.layer_stats)
                layer_event = LayerActivityEvent(
                    id=next_id(),
                    seq=seq - 1,
                    type="LAYER_ACTIVITY",
                    t=t,
                    level="DERIVED",
                    position=prompt_len + step,
                    step=step,
                    layers=[
                        LayerStat(
                            layer=layer,
                            l2Norm=round(float(l2), 3),
                            normRatio=round(float(ratio), 3),
                        )
                        for (layer, l2), ratio in zip(res.layer_stats, ratios)
                    ],
                )
                yield await emit(layer_event)

            if res.attention is not None:
                for layer_idx in sorted(res.attention):
                    row = res.attention[layer_idx]
                    attention_event = AttentionEvent(
                        id=next_id(),
                        seq=seq - 1,
                        type="ATTENTION",
                        t=t,
                        level="DERIVED",
                        position=prompt_len + step,
                        layer=layer_idx + 1,
                        aggregated=[
                            {
                                "position": j - 1,
                                "text": ctx_texts[j],
                                "weight": round(float(w), 4),
                            }
                            for j, w in enumerate(row)
                        ],
                        headEntropyBits=(
                            [
                                round(float(h), 3)
                                for h in res.head_entropies[layer_idx]
                            ]
                            if res.head_entropies is not None
                            else None
                        ),
                    )
                    yield await emit(attention_event)

            if scorer is not None:
                # exact mass the FULL distribution places on each concept's
                # token set; only concepts above threshold become events
                active = [
                    s for s in scorer.score(res.full_log_probs) if s.mass >= CONCEPT_ACTIVE_MASS
                ]
                active.sort(key=lambda s: -s.mass)
                for scored in active:
                    concept_event = ConceptEvent(
                        id=next_id(),
                        seq=seq - 1,
                        type="CONCEPT",
                        t=t,
                        level="INTERPRETED",
                        conceptId=scored.spec.conceptId,
                        label=scored.spec.label,
                        score=round(scored.mass, 4),
                        positions=[prompt_len + step],
                        evidence=[
                            {
                                "tokenId": token_id_,
                                "text": text_,
                                "probability": round(p_, 4),
                            }
                            for token_id_, text_, p_ in scored.evidence
                        ],
                    )
                    yield await emit(concept_event)

            ctx.append(token_id)
            ctx_texts.append(tok.text)
            if backend.is_eos(token_id):
                finish_reason = "stop"
                break

        output_text = _compose_output_text(emitted)
        duration_ms = int((time.perf_counter() - started) * 1000)

        decision = DecisionEvent(
            id=next_id(),
            seq=seq - 1,
            type="DECISION",
            t=t + 4,
            level="DERIVED",
            decision=finish_reason if finish_reason in ("stop", "max_tokens", "aborted") else "max_tokens",
            detail={
                "stop": "end-of-sequence token reached",
                "max_tokens": f"reached max_tokens={max_tokens}",
                "context_limit": f"reached context limit {settings.max_context}",
            }[finish_reason],
        )
        yield await emit(decision)

        output_event = OutputEvent(
            id=next_id(),
            seq=seq - 1,
            type="OUTPUT",
            t=t + 8,
            level="MEASURED",
            text=output_text,
            tokenCount=len(emitted),
            durationMs=duration_ms,
            finishReason=finish_reason,
        )
        yield await emit(output_event)

        if writer is not None:
            await writer.close_trace(
                trace_id,
                output_text=output_text,
                token_count=len(emitted),
                duration_ms=duration_ms,
                status="complete",
            )

        yield format_sse(
            "done",
            json.dumps(
                {
                    "traceId": trace_id,
                    "tokenCount": len(emitted),
                    "durationMs": duration_ms,
                    "finishReason": finish_reason,
                }
            ),
        )

    except GenerationError as exc:
        if writer is not None:
            await writer.record_error(trace_id, exc.message)
        yield format_sse("error", json.dumps({"code": exc.code, "message": exc.message}))
    except Exception as exc:  # noqa: BLE001 — the stream must terminate cleanly
        if writer is not None:
            await writer.record_error(trace_id, str(exc))
        yield format_sse(
            "error", json.dumps({"code": "generation_failed", "message": str(exc)})
        )
