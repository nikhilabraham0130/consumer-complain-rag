"""
Unit Tests for Information Retrieval Metrics (MRR@K, nDCG@K, Recall@K).
"""

import math
import pytest

from evals.metrics import (
    dcg_at_k,
    mean_score,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


# =============================================================================
# Reciprocal Rank (RR@K) Tests
# =============================================================================

def test_reciprocal_rank_rank_one() -> None:
    """Best case: first document retrieved is relevant -> RR = 1.0."""
    retrieved = ["docA", "docB", "docC"]
    positives = {"docA"}
    assert reciprocal_rank(retrieved, positives, k=10) == 1.0


def test_reciprocal_rank_rank_two() -> None:
    """First relevant document is at rank 2 -> RR = 0.5."""
    retrieved = ["noise", "docA", "docB"]
    positives = {"docA", "docB"}
    assert reciprocal_rank(retrieved, positives, k=10) == 0.5


def test_reciprocal_rank_rank_four() -> None:
    """First relevant document is at rank 4 -> RR = 0.25."""
    retrieved = ["noise1", "noise2", "noise3", "docA"]
    positives = {"docA"}
    assert reciprocal_rank(retrieved, positives, k=10) == 0.25


def test_reciprocal_rank_miss_or_cutoff() -> None:
    """Relevant document is past cutoff K=3 -> RR = 0.0."""
    retrieved = ["noise1", "noise2", "noise3", "docA"]
    positives = {"docA"}
    assert reciprocal_rank(retrieved, positives, k=3) == 0.0


def test_reciprocal_rank_empty_positives() -> None:
    """Empty ground truth returns 0.0."""
    assert reciprocal_rank(["docA"], set(), k=10) == 0.0


# =============================================================================
# Recall@K Tests
# =============================================================================

def test_recall_at_k_full() -> None:
    """All known positives retrieved -> Recall = 1.0."""
    retrieved = ["doc1", "doc2", "noise"]
    positives = {"doc1", "doc2"}
    assert recall_at_k(retrieved, positives, k=10) == 1.0


def test_recall_at_k_partial() -> None:
    """2 out of 4 known positives retrieved -> Recall = 0.5."""
    retrieved = ["doc1", "noise1", "doc2", "noise2"]
    positives = {"doc1", "doc2", "doc3", "doc4"}
    assert recall_at_k(retrieved, positives, k=10) == 0.5


def test_recall_at_k_zero() -> None:
    """No known positives retrieved -> Recall = 0.0."""
    retrieved = ["noise1", "noise2"]
    positives = {"doc1", "doc2"}
    assert recall_at_k(retrieved, positives, k=10) == 0.0


# =============================================================================
# nDCG@K Tests
# =============================================================================

def test_ndcg_perfect_order() -> None:
    """Perfect ranking order (2s then 1s then 0s) returns exactly 1.0."""
    retrieved = ["docA", "docB", "docC", "docD"]
    ground_truth = {
        "docA": 2,
        "docB": 2,
        "docC": 1,
        "docD": 0,
    }
    assert ndcg_at_k(retrieved, ground_truth, k=4) == 1.0


def test_ndcg_suboptimal_order() -> None:
    """Suboptimal order (noise ranked before bullseye) returns strictly < 1.0."""
    retrieved = ["docD", "docC", "docB", "docA"]  # Inverted order: 0, 1, 2, 2
    ground_truth = {
        "docA": 2,
        "docB": 2,
        "docC": 1,
        "docD": 0,
    }
    score = ndcg_at_k(retrieved, ground_truth, k=4)
    assert 0.0 < score < 1.0


def test_ndcg_empty_or_zero_truth() -> None:
    """If ground truth has only zeros or is empty, nDCG is 0.0."""
    retrieved = ["docA", "docB"]
    assert ndcg_at_k(retrieved, {}, k=10) == 0.0
    assert ndcg_at_k(retrieved, {"docA": 0, "docB": 0}, k=10) == 0.0


# =============================================================================
# Mean Score Tests
# =============================================================================

def test_mean_score() -> None:
    assert mean_score([0.5, 1.0, 0.75]) == 0.75
    assert mean_score([]) == 0.0
