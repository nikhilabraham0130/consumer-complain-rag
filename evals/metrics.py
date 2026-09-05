"""
Evaluation Metrics for Information Retrieval (IR) and Search Benchmarking.

Implements standard scientific IR ranking metrics:
- Reciprocal Rank (RR@K) -> Mean Reciprocal Rank (MRR@K)
- Normalized Discounted Cumulative Gain (nDCG@K) using graded relevance (2, 1, 0)
- Recall@K
"""

import math
from typing import Dict, List, Sequence, Set


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    ground_truth_positives: Set[str],
    k: int = 10,
) -> float:
    """
    Computes Reciprocal Rank at K (RR@K).

    Looks for the 1-based rank of the FIRST strictly relevant document in the top K.
    If found at rank r <= k, returns 1.0 / r.
    If no relevant document is found in top K, returns 0.0.

    Args:
        retrieved_ids: Ordered list of document IDs returned by the search engine.
        ground_truth_positives: Set of document IDs verified as relevant.
        k: Maximum rank depth to inspect (default: 10).
    """
    if not ground_truth_positives:
        return 0.0

    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in ground_truth_positives:
            return 1.0 / rank

    return 0.0


def recall_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_positives: Set[str],
    k: int = 50,
) -> float:
    """
    Computes Recall at K.

    Formula: (Number of relevant documents in top K) / (Total relevant documents in ground truth)

    Args:
        retrieved_ids: Ordered list of document IDs returned by the search engine.
        ground_truth_positives: Set of document IDs verified as relevant.
        k: Maximum rank depth to inspect (default: 50).
    """
    if not ground_truth_positives:
        return 0.0

    top_k_set = set(retrieved_ids[:k])
    relevant_retrieved = len(top_k_set.intersection(ground_truth_positives))

    return relevant_retrieved / len(ground_truth_positives)


def dcg_at_k(
    scores: Sequence[int],
    k: int = 10,
) -> float:
    """
    Computes Discounted Cumulative Gain at K using the standard exponential gain formulation:
        DCG@K = sum_{i=1}^K (2^{rel_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    for i, rel in enumerate(scores[:k], start=1):
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 1.0)
    return dcg


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_scores: Dict[str, int],
    k: int = 10,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (nDCG@K).

    Evaluates whether the search engine ranked higher-relevance documents (e.g. 2s)
    ahead of lower-relevance documents (1s) and irrelevant noise (0s).

    Args:
        retrieved_ids: Ordered list of document IDs returned by the search engine.
        ground_truth_scores: Mapping of doc_id -> relevance grade (e.g. 2 = relevant, 1 = partial, 0 = non-rel).
        k: Cutoff depth (default: 10).
    """
    if not ground_truth_scores:
        return 0.0

    # 1. Calculate Actual DCG
    actual_scores = [ground_truth_scores.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    actual_dcg = dcg_at_k(actual_scores, k=k)

    # 2. Calculate Ideal DCG (sort all known ground truth scores descending)
    ideal_scores = sorted(ground_truth_scores.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k=k)

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def mean_score(scores: Sequence[float]) -> float:
    """Calculates arithmetic mean, returning 0.0 for empty sequences."""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
