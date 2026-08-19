"""The uncertainty layer's measurable half (spec §22) — answer stability
under controlled prompt perturbations.

Mirrors packages/trace-schema/src/uncertainty.ts 1:1 (same perturbations,
same FNV-1a text seed, same swap pool and rate) so the FakeBackend's
perturbed continuations and the fixture generator's stay identical.

The two quantities this instrument deliberately does NOT measure
(input ambiguity — needs an auxiliary model; evidence quality — needs
retrieval sources) are emitted as null-valued UNCERTAINTY events with
the reason in `basis`; see app/engine/generate.py.
"""

from __future__ import annotations

_M32 = 0xFFFFFFFF

# the four quantities in spec §22's display order — emission and UI both
# render in this order. Mirrored by UNCERTAINTY_KINDS in uncertainty.ts.
UNCERTAINTY_KINDS = [
    "MODEL_UNCERTAINTY",
    "EVIDENCE_QUALITY",
    "INPUT_AMBIGUITY",
    "ANSWER_STABILITY",
]


def prompt_perturbations(text: str) -> list[tuple[str, str]]:
    """Deterministic surface perturbations of a prompt.

    Each entry is (name, perturbed_text). Only perturbations that APPLY to
    this prompt are returned (a lowercase prompt with no final punctuation
    yields none). Mirrored by promptPerturbations() in uncertainty.ts.
    """
    out: list[tuple[str, str]] = []
    trimmed = text.rstrip()
    if len(trimmed) > 1 and trimmed[-1] in ".?!":
        out.append(("strip_final_punct", trimmed[:-1]))
    if text[:1].isupper():
        out.append(("lowercase_first", text[:1].lower() + text[1:]))
    return out


def text_seed(text: str) -> int:
    """FNV-1a over the full string — the seed for one fake perturbed run."""
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & _M32
    return h
