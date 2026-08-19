"""The uncertainty layer's mechanics (spec §22): deterministic prompt
perturbations, the fake's seeded sensitivity model, and the invariants that
make per-position agreement well defined.
"""

from __future__ import annotations

from app.aggregation.stability import prompt_perturbations, text_seed
from app.models.fake_backend import _GENERIC_RESPONSE, _RESPONSES, _perturb_response, split_tokens

SKY_BLUE = "Why is the sky blue?"


def test_perturbations_apply_only_when_meaningful() -> None:
    assert prompt_perturbations(SKY_BLUE) == [
        ("strip_final_punct", "Why is the sky blue"),
        ("lowercase_first", "why is the sky blue?"),
    ]
    # lowercase prompt without punctuation: nothing to perturb
    assert prompt_perturbations("hello world") == []
    # a lone punctuation mark is not a prompt worth perturbing
    assert prompt_perturbations("?") == []
    # case-only and punctuation-only both apply independently
    assert prompt_perturbations("Hello") == [("lowercase_first", "hello")]
    assert prompt_perturbations("wow.") == [("strip_final_punct", "wow")]


def test_text_seed_is_stable_and_discriminating() -> None:
    assert text_seed(SKY_BLUE) == text_seed(SKY_BLUE)
    assert text_seed(SKY_BLUE) != text_seed("why is the sky blue?")
    assert 0 <= text_seed(SKY_BLUE) < 2**32


def test_perturbed_response_preserves_token_count() -> None:
    response = _RESPONSES[0][1]
    perturbed = _perturb_response(response, text_seed("why is the sky blue?"))
    # swaps are word-for-word — per-position agreement stays well defined
    assert len(split_tokens(perturbed)) == len(split_tokens(response))
    # deterministic: same seed, same divergence
    assert perturbed == _perturb_response(response, text_seed("why is the sky blue?"))
    # different seed, different (or at least independently drawn) divergence
    other = _perturb_response(response, text_seed("Why is the sky blue"))
    assert len(split_tokens(other)) == len(split_tokens(response))


def test_perturbed_response_swaps_only_qualifying_words() -> None:
    response = "The quick brown fox jumps over the lazy dog."
    perturbed = _perturb_response(response, seed=1234)
    original_tokens = split_tokens(response)
    perturbed_tokens = split_tokens(perturbed)
    for (orig, _), (new, _) in zip(original_tokens, perturbed_tokens):
        if orig != new:
            # a swapped word came from the pool (len > 2, not punctuation)
            assert len(new) > 2 and new.isalpha()
        else:
            pass  # survived this seed's draw — also fine


def test_fake_continuation_is_verbatim_for_authored_prompts() -> None:
    """The authored prompt reproduces the authored response EXACTLY — the
    sensitivity model never touches the canonical path, so pre-uncertainty
    traces stay byte-identical."""
    from app.models.fake_backend import FakeBackend
    from app.models.registry import MODEL_REGISTRY

    backend = FakeBackend(MODEL_REGISTRY["fake"])
    assert backend._choose_continuation(SKY_BLUE) == _RESPONSES[0][1]
    # canonical variants match the same response but diverge deterministically
    lower = backend._choose_continuation("why is the sky blue?")
    assert lower != _RESPONSES[0][1]
    assert lower == backend._choose_continuation("why is the sky blue?")
    stripped = backend._choose_continuation("Why is the sky blue")
    assert stripped != _RESPONSES[0][1]
    assert len(split_tokens(stripped)) == len(split_tokens(_RESPONSES[0][1]))


def test_generic_prompts_stay_deterministic() -> None:
    """Sensitivity is modeled on authored prompts only; arbitrary prompts
    reproduce the generic response regardless of surface form — a real
    model is the only source of real sensitivity."""
    from app.models.fake_backend import FakeBackend
    from app.models.registry import MODEL_REGISTRY

    backend = FakeBackend(MODEL_REGISTRY["fake"])
    assert backend._choose_continuation("some arbitrary prompt") == _GENERIC_RESPONSE
    assert backend._choose_continuation("Some arbitrary prompt") == _GENERIC_RESPONSE
