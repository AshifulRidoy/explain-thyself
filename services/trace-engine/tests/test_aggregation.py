"""Aggregation: layer stats, top-k, sampling determinism."""

from __future__ import annotations

import numpy as np

from app.aggregation.stats import (
    RunningNormalizer,
    sample_greedy,
    sample_with_temperature,
    top_k_from_log_probs,
)


def test_running_normalizer_first_step_is_one() -> None:
    n = RunningNormalizer(3)
    ratios = n.update([(1, 10.0), (2, 20.0), (3, 30.0)])
    assert list(ratios) == [1.0, 1.0, 1.0]


def test_running_normalizer_converges_to_ratio_mean() -> None:
    n = RunningNormalizer(1)
    ratios = [n.update([(1, v)])[0] for v in [10.0, 10.0, 10.0, 20.0]]
    assert ratios[0] == 1.0
    assert ratios[1] == 1.0  # constant series stays at 1
    assert ratios[3] == 20.0 / 12.5  # mean(10,10,10,20) = 12.5


def test_top_k_descending() -> None:
    lp = np.linspace(-5, 0, 100)
    top = top_k_from_log_probs(lp, k=8)
    assert len(top) == 8
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True)
    assert top[0][0] == 99  # highest log-prob index


def test_greedy_is_deterministic_argmax() -> None:
    lp = np.array([-3.0, -0.5, -1.0, -0.5])
    assert sample_greedy(lp) == 1  # ties broken by lowest id


def test_temperature_sampling_seeded_reproducible() -> None:
    lp = np.array([-1.0, -0.1, -2.0, -0.3])
    a = sample_with_temperature(lp, 0.9, None, np.random.default_rng(7))
    b = sample_with_temperature(lp, 0.9, None, np.random.default_rng(7))
    assert a == b
