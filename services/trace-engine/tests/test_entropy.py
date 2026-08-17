"""Entropy math anchors (docs/interpretability.md)."""

from __future__ import annotations

import numpy as np
import pytest

from app.aggregation.stats import entropy_from_log_probs

GPT2_VOCAB = 50257


def test_delta_distribution_is_zero() -> None:
    lp = np.full(100, -1e9, dtype=np.float64)
    lp[7] = 0.0
    assert entropy_from_log_probs(lp) == pytest.approx(0.0, abs=1e-9)


def test_uniform_over_gpt2_vocab() -> None:
    # uniform p = 1/V ⇒ log-probs all equal -ln(V)
    lp = np.full(GPT2_VOCAB, -np.log(GPT2_VOCAB), dtype=np.float64)
    assert entropy_from_log_probs(lp) == pytest.approx(np.log2(GPT2_VOCAB), rel=1e-12)
    assert entropy_from_log_probs(lp) == pytest.approx(15.6170, abs=1e-4)


def test_hand_computed_four_symbol() -> None:
    # p = [1/2, 1/4, 1/8, 1/8] → H = 1.75 bits exactly
    p = np.array([0.5, 0.25, 0.125, 0.125])
    lp = np.log(p)
    assert entropy_from_log_probs(lp) == pytest.approx(1.75, abs=1e-12)


def test_log_softmax_form_matches_naive_form() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=1000).astype(np.float64)
    lp = logits - np.log(np.exp(logits).sum())
    naive = -(np.exp(lp) * lp).sum() / np.log(2)
    assert entropy_from_log_probs(lp) == pytest.approx(naive, rel=1e-12)


def test_extreme_negative_log_probs_do_not_nan() -> None:
    lp = np.full(10, -np.inf, dtype=np.float64)
    lp[3] = 0.0
    assert entropy_from_log_probs(lp) == pytest.approx(0.0, abs=1e-9)
