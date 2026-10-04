"""Top-K ranking metrics for binary relevance.

`recommended` is a ranked list of unique item ids (best first); `relevant` is the set of held-out
items the user actually interacted with. Precision@K divides by K even if fewer than K items
were returned (a model that can't fill the list is penalised, not rewarded).
"""
from __future__ import annotations

import math
from typing import Collection, Sequence


def _check(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    _check(k)
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    _check(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """AP@K = (1 / min(|relevant|, K)) * sum_{i<=K} P@i * rel_i.  MAP@K is its mean over users."""
    _check(k)
    hits, total = 0, 0.0
    for i, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            total += hits / i
    denom = min(len(relevant), k)
    return total / denom if denom else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Binary-gain NDCG@K with log2 discount."""
    _check(k)
    dcg = sum(1.0 / math.log2(i + 1) for i, item in enumerate(recommended[:k], start=1) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "map": average_precision_at_k,
    "ndcg": ndcg_at_k,
}
