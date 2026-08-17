"""Deterministic, torch-free backend — test double AND offline demo mode.

Mirrors the realism math of packages/trace-schema/src/generate.ts (same
mulberry32 stream, same entropy/probability/top-k/layer-norm rules), so the
full HTTP/SSE/persistence pipeline is testable with zero torch, and the app
demos offline via ETS_MODEL=fake. Not scaffolding to delete.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .backend import StepResult, TopToken
from .registry import ModelSpec

# --- mulberry32: bit-for-bit port of the TypeScript generator --------------

_M32 = 0xFFFFFFFF


def mulberry32(seed: int) -> Callable[[], float]:
    state = seed & _M32

    def rand() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & _M32
        a = state
        t = ((a ^ (a >> 15)) * (a | 1)) & _M32
        t = ((t + ((t ^ (t >> 7)) * (t | 61) & _M32)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0

    return rand


def _word_to_token_id(word: str) -> int:
    h = 2166136261
    for ch in word:
        h ^= ord(ch)
        h = (h * 16777619) & _M32
    return 1000 + (h % 49000)


FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "you", "your", "we", "our", "they",
    "their", "he", "she", "his", "her", "i", "if", "then", "than", "so", "not",
    "no", "can", "will", "would", "should", "may", "might", "do", "does", "did",
    "has", "have", "had", "into", "about", "over", "under", "more", "most",
    "some", "any", "each", "which", "who", "when", "where", "how", "what",
}

_PUNCT = ".,;:?!"


def split_tokens(text: str) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    for piece in text.split(" "):
        if not piece:
            continue
        leading = len(out) > 0
        if piece[-1] in _PUNCT and len(piece) > 1:
            out.append((piece[:-1], leading))
            out.append((piece[-1], False))
        else:
            out.append((piece, leading))
    return out


def _classify(text: str) -> str:
    if all(c in _PUNCT for c in text):
        return "punct"
    if text.lower() in FUNCTION_WORDS:
        return "function"
    return "content"


def _layer_norm_curve(layer: int, layer_count: int, step_gain: float) -> float:
    frac = layer / layer_count
    return (2.5 + 2.4 * layer + 6 * frac * frac) * step_gain


_FAKE_VOCAB = 4096

_RESPONSES: list[tuple[str, str]] = [
    (
        "Why is the sky blue?",
        "The sky looks blue because of how air molecules scatter sunlight. "
        "Sunlight contains all colors, and short blue wavelengths scatter more "
        "strongly than long red ones when they strike particles in the "
        "atmosphere. This effect is called Rayleigh scattering. The scattered "
        "blue light reaches your eyes from every direction, so the whole sky "
        "glows blue. At sunset light travels a longer path, most blue is "
        "scattered away, and the sky turns red and orange.",
    ),
    (
        "Should I learn Python or Rust?",
        "For most people starting out, Python is the better first language. "
        "The ecosystem is enormous, the syntax reads almost like English, and "
        "you can move from small scripts to data work to web services without "
        "switching tools. Rust asks more of you up front: ownership, borrowing, "
        "and lifetimes are real concepts you must understand before the "
        "compiler becomes your friend rather than your adversary. That rigor "
        "pays off in systems programming, where control over memory and "
        "performance matters. A reasonable path is Python first, build real "
        "things, then learn Rust when a project genuinely demands it.",
    ),
]

_GENERIC_RESPONSE = (
    "This is a deterministic offline response from the fake backend, shaped "
    "exactly like a real trace: tokens carry probabilities, entropy, and "
    "layer activity, so the full pipeline can be exercised without a model."
)


class FakeBackend:
    """Deterministic backend; a fixed seed reproduces a trace exactly."""

    spec: ModelSpec

    def __init__(self, spec: ModelSpec, seed: int = 4242) -> None:
        self.spec = spec
        self._rng = mulberry32(seed)
        self._eos_id = _word_to_token_id("<eos>")
        self._prompt_len = 0
        self._continuation: list[tuple[str, bool]] = []
        # running per-layer mean for normRatio (mirrors the TS generator)
        self._layer_means = np.zeros(spec.layer_count)
        self._steps = 0
        self._vocab_pool = [
            "approach", "context", "detail", "structure", "process", "signal",
            "pattern", "framework", "consider", "suggest", "provide", "reflect",
        ]

    def load(self, device: str) -> None:
        # nothing to load — torch-free by design
        pass

    def _choose_continuation(self, prompt: str) -> str:
        for known_prompt, response in _RESPONSES:
            if prompt.strip().lower() == known_prompt.lower():
                return response
        return _GENERIC_RESPONSE

    def encode(self, text: str) -> list[int]:
        prompt_tokens = split_tokens(text)
        self._prompt_len = len(prompt_tokens)
        self._continuation = split_tokens(self._choose_continuation(text))
        topical = [
            w.lower()
            for w, _ in self._continuation
            if len(w) > 2 and w.lower() not in FUNCTION_WORDS
        ]
        self._vocab_pool = list(dict.fromkeys(topical + self._vocab_pool))
        return [_word_to_token_id(w) for w, _ in prompt_tokens] or [self._eos_id]

    def prompt_tokens(self, text: str) -> list[tuple[int, str]]:
        return [(_word_to_token_id(w), w) for w, _ in split_tokens(text)]

    def decode_token(self, token_id: int) -> TopToken:
        # rare path: temperature sampling picked outside the modeled top-8
        return TopToken(
            token_id=token_id,
            text=f"[{token_id}]",
            raw_text=f"Ġ[{token_id}]",
            leading_space=True,
            probability=1e-6,
            rank=99,
        )

    def step(self, ctx: list[int], collect_layers: bool) -> StepResult:
        t0 = time.perf_counter()
        step = len(ctx) - self._prompt_len
        exhausted = step >= len(self._continuation)
        text, leading = (
            self._continuation[step] if not exhausted else ("<eos>", False)
        )

        kind = "punct" if exhausted else _classify(text)
        r = self._rng
        if kind == "punct":
            entropy = 0.05 + r() * 0.3
        elif kind == "function":
            entropy = 0.1 + r() * 0.8
        elif r() < 0.18:
            entropy = 3.5 + r() * 3.0
        else:
            entropy = 1.2 + r() * 2.0

        probability = min(0.995, max(0.02, 2.0 ** (-entropy) * (0.85 + r() * 0.3)))
        decay = min(0.85, max(0.05, 2.0 ** (-1.0 / max(entropy, 0.4))))

        top_k: list[TopToken] = []
        emitted_id = self._eos_id if exhausted else _word_to_token_id(text)
        top_k.append(
            TopToken(
                token_id=emitted_id,
                text=text,
                raw_text=("Ġ" if leading else "") + text,
                leading_space=leading,
                probability=round(probability, 4),
                rank=0,
            )
        )
        p = probability
        alts = [w for w in self._vocab_pool if w != text.lower()]
        for i in range(7):
            p *= decay
            alt = alts[i % len(alts)] if alts else "pattern"
            top_k.append(
                TopToken(
                    token_id=_word_to_token_id(alt),
                    text=alt,
                    raw_text="Ġ" + alt,
                    leading_space=True,
                    probability=round(p, 4),
                    rank=i + 1,
                )
            )

        # sparse fake distribution over the fake vocab: only the 8 candidates
        # have mass; entropy computed downstream from this array is self-consistent
        log_probs = np.full(_FAKE_VOCAB, -60.0, dtype=np.float32)
        for cand in top_k:
            log_probs[cand.token_id % _FAKE_VOCAB] = np.log(max(cand.probability, 1e-12))

        layer_stats: list[tuple[int, float]] | None = None
        if collect_layers:
            step_gain = 0.9 + r() * 0.25
            xs = [
                round(_layer_norm_curve(i + 1, self.spec.layer_count, step_gain), 3)
                for i in range(self.spec.layer_count)
            ]
            x = np.array(xs)
            self._steps += 1
            self._layer_means += (x - self._layer_means) / self._steps
            layer_stats = [(i + 1, float(v)) for i, v in enumerate(x)]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        # report the modeled latency, not the near-zero real one
        latency_ms = round(38 + r() * 45 + (step % 4 == 0 and 25 or 0), 1)

        return StepResult(
            top_k=top_k,
            full_log_probs=log_probs,
            layer_stats=layer_stats,
            latency_ms=latency_ms,
        )

    def is_eos(self, token_id: int) -> bool:
        return token_id == self._eos_id
