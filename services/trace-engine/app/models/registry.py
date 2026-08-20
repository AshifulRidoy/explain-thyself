"""Model registry — swapping the model under the microscope is config-level.

Adding Qwen2.5-0.5B or Gemma later is ONE ModelSpec line here (verify the
family is supported by the installed TransformerLens release at swap time)
plus a smoke test. Nothing else in the engine changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    # TransformerLens model name, or None for torch-free backends (fake)
    tl_name: str | None
    hf_id: str
    layer_count: int
    head_count: int
    d_model: int
    param_count: int
    dtype: str
    emits_resid_activity: bool
    # Tokenizer identity: models that share a tokenizer produce comparable
    # token IDS, which is the precondition for cross-model token agreement
    # (spec Phase 7 comparison). "gpt2-bpe" = the 50257-entry GPT-2 BPE
    # shared by gpt2 and its distillations; the fake's ids are word hashes
    # in the same numeric range — comparable only with itself.
    tokenizer: str = "gpt2-bpe"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gpt2-small": ModelSpec(
        key="gpt2-small",
        # official transformers name since TL v3 ("gpt2-small" is a
        # deprecated alias that warns on every load)
        tl_name="gpt2",
        hf_id="gpt2",
        layer_count=12,
        head_count=12,
        d_model=768,
        param_count=124_000_000,
        dtype="float32",
        emits_resid_activity=True,
        tokenizer="gpt2-bpe",
    ),
    # Distilled GPT-2 (82M): the second microscope slide. Same tokenizer
    # and d_model=768 as gpt2-small, so token agreement is well-defined
    # and the search embedding column stays valid. Dims verified against
    # the loaded checkpoint (6 layers, 12 heads, d_vocab 50257).
    "distilgpt2": ModelSpec(
        key="distilgpt2",
        tl_name="distilgpt2",
        hf_id="distilgpt2",
        layer_count=6,
        head_count=12,
        d_model=768,
        param_count=82_000_000,
        dtype="float32",
        emits_resid_activity=True,
        tokenizer="gpt2-bpe",
    ),
    # Deterministic, torch-free backend: test double + offline demo mode.
    # Signals imitate gpt2-small shapes (revision "fixture" marks provenance).
    "fake": ModelSpec(
        key="fake",
        tl_name=None,
        hf_id="fake",
        layer_count=12,
        head_count=12,
        d_model=768,
        param_count=124_000_000,
        dtype="float32",
        emits_resid_activity=True,
        tokenizer="fake-wordhash",
    ),
}


@dataclass
class BackendStatus:
    device: str = "unloaded"
    loaded: bool = False
    # pass | skipped | cpu_fallback | unloaded
    self_check: str = "unloaded"


# one status per loaded model: comparison (spec Phase 7) keeps two real
# backends resident at once, and /health must stay honest about WHICH
# model it is describing
_status: dict[str, BackendStatus] = {}


def backend_status(key: str | None = None) -> BackendStatus:
    """Status of one model's backend; `key=None` means the active model."""
    from ..config import settings

    return _status.get(key or settings.model, BackendStatus())


def set_backend_status(key: str, **changes: str) -> None:
    current = _status.get(key, BackendStatus())
    _status[key] = BackendStatus(**{**current.__dict__, **changes})
