"""TransformerLens backend — real model internals (spec Phase 2–3).

Instrumentation rules (docs/interpretability.md):
  - BASIC mode caches NOTHING (logits only)
  - STANDARD caches exactly blocks.{i}.hook_resid_post, i = 0..L-1
  - RESEARCH adds attention, reduced HOOK-SIDE inside run_with_hooks
    (head-mean of the final query row); the [heads, q, k] pattern tensor
    never leaves the hook and no cache is built at all
  - extraction reads ONLY the final position; the cache never leaves step()
  - the cache reference is dropped every step — a retained ActivationCache
    is an OOM waiting to happen
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

        self.device = resolve_device(device)
        # v3 path: the bridge boots raw HF weights; compatibility mode gives
        # HookedTransformer-equivalent processing + legacy hook names
        # (blocks.{i}.hook_resid_post). Verified to match HF reference logits
        # exactly (see tests/test_real_model.py).
        from transformer_lens.model_bridge import TransformerBridge

        bridge = TransformerBridge.boot_transformers(
            self.spec.tl_name,
            dtype=torch.float32,
            device=self.device,
        )
        bridge.enable_compatibility_mode(disable_warnings=True)
        self.model = bridge

    def numerics_self_check(self) -> bool:
        """Run a fixed prompt on the loaded device and on CPU; compare
        top-1 token and max |logit diff|. Mismatch ⇒ the device is lying —
        the caller should reload on CPU.

        Returns True when the loaded device passes (or is already CPU).
        """
        import torch

        if self.model is None or self.device == "cpu":
            return True
        from transformer_lens.model_bridge import TransformerBridge

        with torch.no_grad():
            ids = self.model.to_tokens(_CALIBRATION_PROMPT, prepend_bos=True)
            device_logits = self.model(ids)[0, -1].float().cpu()

            cpu_model = TransformerBridge.boot_transformers(
                self.spec.tl_name, dtype=torch.float32, device="cpu"
            )
            cpu_model.enable_compatibility_mode(disable_warnings=True)
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

    @property
    def vocab_size(self) -> int:
        return int(self.model.cfg.d_vocab)

    def prompt_tokens(self, text: str) -> list[tuple[int, str]]:
        ids = self.model.to_tokens(text, prepend_bos=False)[0].tolist()
        return [(int(tid), self._decode(tid).text) for tid in ids]

    def decode_token(self, token_id: int) -> TopToken:
        return self._decode(token_id)

    def word_token_ids(self, word: str) -> list[int]:
        """Token ids that can realize `word` in running text: the bare form
        and the leading-space form, each only when it is a SINGLE token.
        Multi-token words return fewer ids (or none) — the scorer then
        measures less mass for that concept, honestly.
        """
        tokenizer = self.model.tokenizer
        bare = tokenizer.encode(word, add_special_tokens=False)
        spaced = tokenizer.encode(" " + word, add_special_tokens=False)
        ids: list[int] = []
        if len(bare) == 1:
            ids.append(int(bare[0]))
        if len(spaced) == 1 and spaced[0] not in ids:
            ids.append(int(spaced[0]))
        return ids

    def token_text(self, token_id: int) -> str:
        return self._decode(token_id).text

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

    def embed_prompt(self, text: str) -> list[float]:
        """The model's own representation of a prompt (spec §28): mean of
        the FINAL layer's resid_post over the prompt's tokens (BOS
        included, matching encode()), L2-normalized — the vector search
        ranks by cosine. One forward pass, cache filtered to a single
        hook, dropped before return: 768 floats leave, no tensors."""
        import torch

        final_hook = f"blocks.{self.spec.layer_count - 1}.hook_resid_post"
        ids = self.model.to_tokens(text, prepend_bos=True)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                ids,
                names_filter=lambda name: name == final_hook,
                prepend_bos=False,  # BOS already prepended above
            )
            resid = cache[final_hook][0].float()  # [seq, d_model]
            mean = resid.mean(dim=0)
            del cache
        vec = mean / mean.norm()
        return [round(v, 6) for v in vec.cpu().tolist()]

    def step(
        self, ctx: list[int], collect_layers: bool, collect_attention: bool = False
    ) -> StepResult:
        import torch

        t0 = time.perf_counter()
        ids = torch.tensor([ctx], dtype=torch.long, device=self.device)

        if collect_attention:
            return self._step_with_hooks(ids, collect_layers, t0)

        filt: Callable[[str], bool] = (
            (lambda name: RESID_POST.match(name) is not None)
            if collect_layers
            else (lambda _name: False)
        )
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                ids,
                names_filter=filt,
                prepend_bos=False,  # BOS already prepended by encode()
            )
            final = logits[0, -1].float()
            log_probs = torch.log_softmax(final, dim=-1)
            top_k = self._top_k(log_probs)

            layer_stats: list[tuple[int, float]] | None = None
            if collect_layers:
                layer_stats = []
                for i in range(self.spec.layer_count):
                    resid = cache[f"blocks.{i}.hook_resid_post"][0, -1].float()
                    layer_stats.append((i + 1, float(resid.norm())))
            # drop the cache before the next step — a retained ActivationCache
            # is an OOM waiting to happen (3.7.2 caches are plain objects,
            # no .clear(); going out of scope frees the tensors)
            del cache

        return StepResult(
            top_k=top_k,
            full_log_probs=log_probs.detach().cpu().numpy().astype(np.float32),
            layer_stats=layer_stats,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _top_k(self, log_probs) -> list[TopToken]:
        """Top-8 tokens + probabilities from final-position log-probs."""
        import torch

        top = torch.topk(log_probs, k=8)
        top_k: list[TopToken] = []
        for rank, (tid, lp) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
            tok = self._decode(tid)
            tok.probability = float(torch.exp(torch.tensor(lp)))
            tok.rank = rank
            top_k.append(tok)
        return top_k

    def _step_with_hooks(self, ids, collect_layers: bool, t0: float) -> StepResult:
        """RESEARCH path — attention is reduced INSIDE the hook.

        `blocks.{i}.attn.hook_pattern` fires with the full
        [batch, heads, q, k] tensor; we take the final query row, mean over
        heads, and move that one [k] vector to CPU immediately. The full
        pattern tensor never survives the hook (spec §19: never cache
        attention). Residual norms are folded into the same pass, so this
        path caches nothing at all — no ActivationCache is ever built.
        """
        import torch

        attention: dict[int, np.ndarray] = {}
        head_entropies: dict[int, list[float]] = {}
        resid_norms: dict[int, float] = {}

        def attn_hook(i: int):
            def hook(tensor, hook=None, *, layer=i):
                final = tensor[0, :, -1, :].float()  # [heads, k]
                attention[layer] = (
                    final.mean(dim=0).detach().cpu().numpy().astype(np.float32)
                )
                p = torch.clamp(final, min=1e-12)
                head_entropies[layer] = (
                    (-(p * torch.log2(p)).sum(dim=-1)).detach().cpu().tolist()
                )

            return hook

        def resid_hook(i: int):
            def hook(tensor, hook=None, *, layer=i):
                resid_norms[layer] = float(tensor[0, -1].float().norm())

            return hook

        fwd_hooks: list[tuple[str, object]] = [
            (f"blocks.{i}.attn.hook_pattern", attn_hook(i))
            for i in range(self.spec.layer_count)
        ]
        if collect_layers:
            fwd_hooks += [
                (f"blocks.{i}.hook_resid_post", resid_hook(i))
                for i in range(self.spec.layer_count)
            ]

        with torch.no_grad():
            logits = self.model.run_with_hooks(
                ids, fwd_hooks=fwd_hooks, prepend_bos=False
            )
            final = logits[0, -1].float()
            log_probs = torch.log_softmax(final, dim=-1)
            top_k = self._top_k(log_probs)

        layer_stats = (
            [(i + 1, resid_norms[i]) for i in range(self.spec.layer_count)]
            if collect_layers
            else None
        )
        return StepResult(
            top_k=top_k,
            full_log_probs=log_probs.detach().cpu().numpy().astype(np.float32),
            layer_stats=layer_stats,
            attention=attention,
            head_entropies=head_entropies,
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
