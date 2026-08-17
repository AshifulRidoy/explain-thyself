"""Aggregation math (spec §13, §18): raw tensors → small honest scalars.

Pure functions only — exhaustively unit-tested (tests/test_entropy.py,
tests/test_aggregation.py). Every number the UI shows traces back to one of
these definitions, documented on /methodology.
"""

from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)


def entropy_from_log_probs(log_probs: np.ndarray) -> float:
    """Shannon entropy in bits of the distribution whose log-probabilities
    (natural log, log-softmax) are given. Full vocab, no top-k truncation.

    Uses the log-sum form -Σ exp(lp)·lp so no p·log p underflow occurs;
    zero-probability entries (lp → -inf or very negative) contribute 0.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    p = np.exp(lp)
    with np.errstate(invalid="ignore"):  # 0 * -inf warns before `where` masks it
        contrib = np.where(p > 0.0, p * lp, 0.0)
    return float(-np.sum(contrib) / LN2)


def probs_from_log_probs(log_probs: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(log_probs, dtype=np.float64))


def top_k_from_log_probs(
    log_probs: np.ndarray, k: int = 8
) -> list[tuple[int, float]]:
    """(token_id, probability) pairs, descending, from full log-probs."""
    lp = np.asarray(log_probs)
    k = min(k, lp.shape[0])
    idx = np.argpartition(lp, -k)[-k:]
    idx = idx[np.argsort(lp[idx])[::-1]]
    p = np.exp(lp[idx].astype(np.float64))
    return [(int(i), float(pi)) for i, pi in zip(idx, p)]


class RunningNormalizer:
    """Welford running mean per layer index.

    normRatio = l2Norm / running_mean_including_current_step, so the first
    step is exactly 1.0 and the value streams honestly without needing a
    global min-max over a finished trace.
    """

    def __init__(self, layer_count: int) -> None:
        self._count = 0
        self._mean = np.zeros(layer_count, dtype=np.float64)

    def update(self, layer_stats: list[tuple[int, float]]) -> np.ndarray:
        """Feed (layer, l2Norm) for one step; returns normRatio per layer."""
        x = np.array([v for _, v in layer_stats], dtype=np.float64)
        self._count += 1
        self._mean += (x - self._mean) / self._count
        return x / self._mean


def sample_greedy(log_probs: np.ndarray) -> int:
    """Argmax with deterministic tie-breaking (lowest token id wins)."""
    lp = np.asarray(log_probs)
    return int(np.argmax(lp))


def sample_with_temperature(
    log_probs: np.ndarray, temperature: float, top_k: int | None, rng: np.random.Generator
) -> int:
    """Temperature/top-k sampling. Used when temperature > 0."""
    lp = np.asarray(log_probs, dtype=np.float64)
    if top_k is not None and top_k > 0:
        k = min(top_k, lp.shape[0])
        cutoff = np.partition(lp, -k)[-k]
        lp = np.where(lp >= cutoff, lp, -np.inf)
    scaled = lp / max(temperature, 1e-6)
    scaled -= scaled.max()
    p = np.exp(scaled)
    p /= p.sum()
    return int(rng.choice(p.shape[0], p=p))
