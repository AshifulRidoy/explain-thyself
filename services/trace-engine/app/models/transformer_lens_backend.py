"""TransformerLens backend — real model internals (spec Phase 2–3).

Instrumentation rules (docs/interpretability.md):
  - BASIC mode caches NOTHING (logits only)
  - STANDARD caches exactly blocks.{i}.hook_resid_post, i = 0..L-1
  - extraction reads ONLY the final position; the cache never leaves step()
  - cache.clear() after every step — a retained ActivationCache is an OOM
  - everything is float32: bf16/f64 are unsupported or unsafe on MPS

Device caveat: TransformerLens has reported silently-wrong GPT-2 outputs on
MPS (TransformerLensOrg/TransformerLens#1178). We therefore self-check the
chosen device against CPU at load time and fall back if numbers disagree.
"""

from __future__ import annotations

import re
import time
from typing import Callable

import numpy as np

from .backend import StepResult, TopToken
from .registry import ModelSpec

RESID_POST = re.compile(r"^blocks\.(\d+)\.hook_resid_post$")

_CALIBRATION_PROMPT = "The capital of France is"


def resolve_device(requested: str) -> str:
    """auto | mps | cpu → concrete torch device string."""
    import torch

    mps_ok = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        return "mps" if mps_ok else "cpu"
    return "mps" if mps_ok else "cpu"


class TransformerLensBackend:
    spec: ModelSpec

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.model = None
        self.device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from transformer_lens import HookedTransformer

        self.device = resolve_device(device)
        self.model = HookedTransformer.from_pretrained(
            self.spec.tl_name,
            dtype=torch.float32,
            device=self.device,
        )
        self.model.eval()

    def numerics_self_check(self) -> bool:
        """Run a fixed prompt on the loaded device and on CPU; compare
        top-1 token and max |logit diff|. Mismatch ⇒ the device is lying —
        the caller should reload on CPU.

        Returns True when the loaded device passes (or is already CPU).
        """
        import torch

        if self.model is None or self.device == "cpu":
            return True
        from transformer_lens import HookedTransformer

        with torch.no_grad():
            ids = self.model.to_tokens(_CALIBRATION_PROMPT, prepend_bos=True)
            device_logits = self.model(ids)[0, -1].float().cpu()

            cpu_model = HookedTransformer.from_pretrained(
                self.spec.tl_name, dtype=torch.float32, device="cpu"
            )
            cpu_model.eval()
            try:
                cpu_logits = cpu_model(ids.to("cpu"))[0, -1].float()
            finally:
                del cpu_model

        same_top1 = int(device_logits.argmax()) == int(cpu_logits.argmax())
        max_diff = float((device_logits - cpu_logits).abs().max())
        return same_top1 and max_diff <= 0.5

    def encode(self, text: str) -> list[int]:
        ids = self.model.to_tokens(text, prepend_bos=True)
        return ids[0].tolist()

    def prompt_tokens(self, text: str) -> list[tuple[int, str]]:
        ids = self.model.to_tokens(text, prepend_bos=False)[0].tolist()
        return [(int(tid), self._decode(tid).text) for tid in ids]

    def decode_token(self, token_id: int) -> TopToken:
        return self._decode(token_id)

    def _decode(self, token_id: int) -> TopToken:
        tokenizer = self.model.tokenizer
        tid = int(token_id)
        raw = tokenizer.convert_ids_to_tokens(tid)
        leading = raw.startswith("Ġ") or raw.startswith("Ċ")
        # single-token decode handles byte-level BPE correctly (partial UTF-8
        # shows as �, which we keep visible rather than hide)
        decoded = tokenizer.decode([tid])
        if leading and decoded[:1] in (" ", "\n"):
            text = decoded[1:]
        else:
            text = decoded
        return TopToken(
            token_id=tid,
            text=text,
            raw_text=raw,
            leading_space=leading,
            probability=0.0,  # filled by step()
            rank=0,
        )

    def step(self, ctx: list[int], collect_layers: bool) -> StepResult:
        import torch

        t0 = time.perf_counter()
        filt: Callable[[str], bool] = (
            (lambda name: RESID_POST.match(name) is not None)
            if collect_layers
            else (lambda _name: False)
        )
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                torch.tensor([ctx], dtype=torch.long, device=self.device),
                names_filter=filt,
                prepend_bos=False,  # BOS already prepended by encode()
            )
            final = logits[0, -1].float()
            log_probs = torch.log_softmax(final, dim=-1)

            top = torch.topk(log_probs, k=8)
            top_k: list[TopToken] = []
            for rank, (tid, lp) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
                tok = self._decode(tid)
                tok.probability = float(torch.exp(torch.tensor(lp)))
                tok.rank = rank
                top_k.append(tok)

            layer_stats: list[tuple[int, float]] | None = None
            if collect_layers:
                layer_stats = []
                for i in range(self.spec.layer_count):
                    resid = cache[f"blocks.{i}.hook_resid_post"][0, -1].float()
                    layer_stats.append((i + 1, float(resid.norm())))
            cache.clear()

        return StepResult(
            top_k=top_k,
            full_log_probs=log_probs.detach().cpu().numpy().astype(np.float32),
            layer_stats=layer_stats,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def is_eos(self, token_id: int) -> bool:
        eos = getattr(self.model, "tokenizer", None)
        if eos is None:
            return False
        # GPT-2 has no dedicated EOS in normal use; use the tokenizer's if present
        eos_id = getattr(self.model, "eos_token_id", None) or getattr(
            eos, "eos_token_id", None
        )
        return eos_id is not None and int(token_id) == int(eos_id)
