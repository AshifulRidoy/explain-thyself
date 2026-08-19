"""The backend seam: everything the generation loop needs from a model.

Two implementations:
  - TransformerLensBackend — real model internals (Phases 2–3)
  - FakeBackend — deterministic, torch-free; powers tests and offline demo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class TopToken:
    token_id: int
    text: str
    raw_text: str
    leading_space: bool
    probability: float
    rank: int


@dataclass
class StepResult:
    top_k: list[TopToken]
    # log-softmax over the full vocab at the final position (np.float32)
    full_log_probs: np.ndarray
    # (layer, l2_norm at final position); None when collect_layers=False
    layer_stats: list[tuple[int, float]] | None
    # layer -> head-mean attention from the final position to every prior
    # position, length == len(ctx); reduced HOOK-SIDE, never a cached
    # [heads, seq, seq] tensor. None when collect_attention=False.
    attention: dict[int, np.ndarray] | None = None
    # layer -> per-head entropy (bits) of that head's final-position row
    head_entropies: dict[int, list[float]] | None = None
    latency_ms: float = 0.0


@runtime_checkable
class ModelBackend(Protocol):
    spec: object  # ModelSpec; typed loosely to avoid a circular import

    # vocabulary size — the denominator of normalized entropy (uncertainty
    # layer, spec §22); real: d_vocab, fake: the fixed fake vocab
    @property
    def vocab_size(self) -> int: ...

    def load(self, device: str) -> None: ...

    def encode(self, text: str) -> list[int]: ...

    def prompt_tokens(self, text: str) -> list[tuple[int, str]]: ...

    def decode_token(self, token_id: int) -> TopToken: ...

    # Phase 5 seam: resolve a concept-dictionary word to the token ids that
    # can realize it (a real tokenizer returns the bare and leading-space
    # variants when each is a single token; empty list = word unmatchable).
    def word_token_ids(self, word: str) -> list[int]: ...

    # Human-readable text for one of those ids (concept evidence rows)
    def token_text(self, token_id: int) -> str: ...

    def step(
        self, ctx: list[int], collect_layers: bool, collect_attention: bool = False
    ) -> StepResult: ...

    def is_eos(self, token_id: int) -> bool: ...
