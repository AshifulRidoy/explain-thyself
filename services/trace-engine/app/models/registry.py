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


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "gpt2-small": ModelSpec(
        key="gpt2-small",
        tl_name="gpt2-small",
        hf_id="gpt2",
        layer_count=12,
        head_count=12,
        d_model=768,
        param_count=124_000_000,
        dtype="float32",
        emits_resid_activity=True,
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
    ),
}


@dataclass
class BackendStatus:
    device: str = "unloaded"
    loaded: bool = False
    # pass | skipped | cpu_fallback | unloaded
    self_check: str = "unloaded"


_status = BackendStatus()


def backend_status() -> BackendStatus:
    return _status


def set_backend_status(**changes: str) -> None:
    global _status
    _status = BackendStatus(**{**_status.__dict__, **changes})
