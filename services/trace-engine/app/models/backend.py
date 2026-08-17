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
    latency_ms: float


@runtime_checkable
class ModelBackend(Protocol):
    spec: object  # ModelSpec; typed loosely to avoid a circular import

    def load(self, device: str) -> None: ...

    def encode(self, text: str) -> list[int]: ...

    def prompt_tokens(self, text: str) -> list[tuple[int, str]]: ...

    def decode_token(self, token_id: int) -> TopToken: ...

    def step(self, ctx: list[int], collect_layers: bool) -> StepResult: ...

    def is_eos(self, token_id: int) -> bool: ...
