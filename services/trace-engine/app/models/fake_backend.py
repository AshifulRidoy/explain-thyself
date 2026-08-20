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

from ..aggregation.concepts import CONCEPT_DICTIONARY
from ..aggregation.stability import text_seed
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


# One slot per possible _word_to_token_id (they live in [1000, 50000)), so
# every dictionary word lands in its OWN slot — the fake id-space stays
# disjoint exactly like the real vocabulary, and concept mass attributes to
# exactly one label.
_FAKE_VOCAB = 50_000

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

# replacement words for perturbed runs — plausible running text, so a
# diverging fake trace still reads like language (mirrored in generate.ts)
_SWAP_POOL = [
    "notably", "broadly", "quickly", "largely", "partly", "slowly",
    "widely", "deeply", "often", "usually", "gently", "readily",
]
_SWAP_RATE = 0.18


def _perturb_response(response: str, seed: int) -> str:
    """A perturbed prompt's fake continuation: the authored response with a
    deterministic seeded subset of content words swapped.

    Draw order is part of the contract (mirrored 1:1 by perturbResponse in
    generate.ts): one rate draw per qualifying token (len > 2, not all
    punctuation), plus a pool draw only when swapping. Token count never
    changes — swaps are word-for-word — so per-position agreement with the
    original run is well defined.
    """
    rng = mulberry32(seed)
    parts: list[str] = []
    for word, leading in split_tokens(response):
        if len(word) > 2 and not all(c in _PUNCT for c in word) and rng() < _SWAP_RATE:
            word = _SWAP_POOL[int(rng() * len(_SWAP_POOL))]
        parts.append((" " if leading else "") + word)
    return "".join(parts)


def _word_edits(a: str, b: str) -> int:
    """Differing word positions between two same-length word lists. Prompts
    of different word counts are infinitely far apart."""
    wa, wb = a.split(), b.split()
    if len(wa) != len(wb):
        return 1 << 30
    return sum(1 for x, y in zip(wa, wb) if x != y)


class FakeBackend:
    """Deterministic backend; a fixed seed reproduces a trace exactly."""

    spec: ModelSpec

    def __init__(self, spec: ModelSpec, seed: int = 4242) -> None:
        self.spec = spec
        self._seed = seed
        self._rng = mulberry32(seed)
        self._eos_id = _word_to_token_id("<eos>")
        self._prompt_len = 0
        self._continuation: list[tuple[str, bool]] = []
        # reverse map for token_text() (concept evidence rows)
        self._id_words: dict[int, str] = {}
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

    @property
    def vocab_size(self) -> int:
        return _FAKE_VOCAB

    def _choose_continuation(self, prompt: str) -> str:
        # Canonical matching (case/punct-insensitive): the authored prompt
        # verbatim reproduces the authored response EXACTLY (existing traces
        # stay byte-identical); a surface perturbation of it models a real
        # model's sensitivity — same response, seeded word swaps at
        # deterministic positions. Mirrored by perturbResponse() in
        # generate.ts.
        canon = prompt.strip().lower().rstrip("?.!").rstrip()
        for known_prompt, response in _RESPONSES:
            if canon == known_prompt.strip().lower().rstrip("?.!").rstrip():
                if prompt == known_prompt:
                    return response
                return _perturb_response(response, text_seed(prompt))
        # A WORD-level edit of an authored prompt (a counterfactual variable,
        # e.g. "How is the sky blue?") is near-miss, not unknown: model it as
        # the seeded-swap variant too — a small edit shifts some tokens, not
        # the whole register. Truly unrelated prompts get the generic
        # response.
        for known_prompt, response in _RESPONSES:
            known = known_prompt.strip().lower().rstrip("?.!").rstrip()
            if _word_edits(canon, known) <= 2:
                return _perturb_response(response, text_seed(prompt))
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

    def word_token_ids(self, word: str) -> list[int]:
        # one pseudo-id per word — always a "single token" in the fake vocab
        token_id = _word_to_token_id(word)
        self._id_words[token_id] = word
        return [token_id]

    def token_text(self, token_id: int) -> str:
        return self._id_words.get(token_id, f"[{token_id}]")

    def embed_prompt(self, text: str) -> list[float]:
        """Deterministic stand-in: the fake has no residual stream to
        represent a prompt with, so search gets a stable seeded unit
        vector — same text always maps to the same vector, which is the
        property the pipeline (storage, SQL, ranking) actually needs
        offline. Never presented as a representation of anything.

        Mean-centered before normalizing: mulberry32 streams from nearby
        seeds share a constant-direction bias (uncentered cosines sit at
        ~0.76 between ANY two texts), which would make the fake rank
        everything equally "similar". Centering leaves unrelated texts
        near 0 and identical texts at exactly 1.0."""
        from ..aggregation.search import EMBEDDING_DIM

        rng = mulberry32(text_seed(text))
        vec = np.array([rng() for _ in range(EMBEDDING_DIM)], dtype=np.float64)
        vec -= vec.mean()
        vec /= np.linalg.norm(vec)
        return [round(float(v), 6) for v in vec]

    def _concept_bump(self, step: int, occupied: np.ndarray) -> dict[int, float]:
        """Deterministic concept mass for this step's fake distribution.

        Separate seeded stream (like _fake_attention's +777_000 offset), so
        concept draws never shift the main randomness. Authored mass is
        pre-renormalization; after dividing by the distribution total it
        stays above CONCEPT_ACTIVE_MASS (0.05). Mirrored 1:1 by the TS
        fixture generator — keep them in sync.
        """
        rng = mulberry32(self._seed + 888_000 + step * 173)
        order = list(range(len(CONCEPT_DICTIONARY)))
        for i in range(len(order) - 1, 0, -1):
            j = int(rng() * (i + 1))
            order[i], order[j] = order[j], order[i]
        n_active = 0 if rng() < 0.10 else (1 if rng() < 0.72 else 2)

        bump: dict[int, float] = {}
        for idx in order[:n_active]:
            words = list(CONCEPT_DICTIONARY[idx].words)
            for k in range(len(words) - 1, 0, -1):
                j = int(rng() * (k + 1))
                words[k], words[j] = words[j], words[k]
            mass = 0.12 + rng() * 0.22
            picked: list[tuple[int, str]] = []
            for word in words:
                if len(picked) >= 3:
                    break
                token_id = _word_to_token_id(word)
                # skip slots already carrying mass (a top-k candidate that
                # IS a dictionary word) — never double-place
                if occupied[token_id] or token_id in bump:
                    continue
                picked.append((token_id, word))
            if picked:  # concept mass == authored mass regardless of skips
                share = mass / len(picked)
                for token_id, word in picked:
                    bump[token_id] = share
                    self._id_words[token_id] = word
        return bump

    def _fake_attention(self, ctx: list[int], step: int):
        """Deterministic imitation of the hook-side reduction: per-layer
        head rows (BOS sink grows with depth, recency decay), softmax
        normalized, averaged — so the mean sums to 1 exactly like the real
        pattern hook. Separate seeded stream: attention draws never shift
        the main randomness, so STANDARD traces stay byte-identical to
        before this existed."""
        attention: dict[int, np.ndarray] = {}
        head_entropies: dict[int, list[float]] = {}
        n = len(ctx)
        for i in range(self.spec.layer_count):
            rng = mulberry32(self._seed + 777_000 + step * 131 + i * 37)
            rows = []
            for _head in range(self.spec.head_count):
                w = np.zeros(n)
                bos_bias = 0.2 + 0.6 * (i + 1) / self.spec.layer_count
                for j in range(n):
                    if j == 0:
                        w[j] = bos_bias + rng() * 0.25
                    else:
                        recency = np.exp(-(n - 1 - j) / 6.0)
                        w[j] = recency * (0.3 + rng() * 0.9)
                w /= w.sum()
                rows.append(w)
            mean = np.mean(rows, axis=0)
            attention[i] = mean.astype(np.float32)
            head_entropies[i] = [
                float(-np.sum(np.where(r > 0, r * np.log2(np.clip(r, 1e-12, 1.0)), 0.0)))
                for r in rows
            ]
        return attention, head_entropies

    def step(
        self, ctx: list[int], collect_layers: bool, collect_attention: bool = False
    ) -> StepResult:
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
        # (plus this step's concept words) have mass. Built as probabilities,
        # renormalized to sum to 1 exactly like a real log-softmax, then
        # logged — entropy downstream is self-consistent by construction.
        probs = np.zeros(_FAKE_VOCAB, dtype=np.float64)
        for cand in top_k:
            slot = cand.token_id % _FAKE_VOCAB
            probs[slot] = max(probs[slot], cand.probability)
        for token_id, mass in self._concept_bump(step, probs > 0).items():
            probs[token_id] += mass
        total = probs.sum()
        probs /= total
        log_probs = np.log(np.clip(probs, 1e-12, 1.0)).astype(np.float32)

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

        attention, head_entropies = (
            self._fake_attention(ctx, step) if collect_attention else (None, None)
        )

        return StepResult(
            top_k=top_k,
            full_log_probs=log_probs,
            layer_stats=layer_stats,
            attention=attention,
            head_entropies=head_entropies,
            latency_ms=latency_ms,
        )

    def is_eos(self, token_id: int) -> bool:
        return token_id == self._eos_id
