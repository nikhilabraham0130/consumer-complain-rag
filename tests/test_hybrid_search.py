"""
Unit tests for RRF fusion, MMR diversity selection, and the full hybrid
search pipeline.

Pure-logic tests (fusion math, MMR math) use no models and run instantly.
Tests needing the cross-encoder share ONE module-scoped instance — it takes
real time to load, so reloading it per test would make this file the
slowest thing in the suite for no reason.
"""

from typing import Any, Dict, List

import pandas as pd
import pytest
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid_search import (
    HybridSearchEngine,
    cross_encoder_rerank,
    mmr_select,
    reciprocal_rank_fusion,
)
from src.retrieval.sparse_index import SparseIndex


# =============================================================================
# reciprocal_rank_fusion() — pure logic, no models
# =============================================================================

def test_rrf_combines_ranks_from_both_lists() -> None:
    dense = [("a", 0.9), ("b", 0.5)]
    sparse = [("b", 5.0), ("c", 3.0)]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    fused_ids = [cid for cid, _score in fused]

    # b: rank 2 in dense (1/62) + rank 1 in sparse (1/61) -> highest combined
    assert fused_ids[0] == "b"
    assert set(fused_ids) == {"a", "b", "c"}


def test_rrf_document_in_both_lists_outranks_single_list_match() -> None:
    dense = [("x", 0.9), ("shared", 0.8)]
    sparse = [("shared", 4.0), ("y", 3.0)]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    fused_ids = [cid for cid, _score in fused]

    assert fused_ids[0] == "shared", "appearing in both lists should beat appearing in only one"


def test_rrf_handles_empty_lists() -> None:
    assert reciprocal_rank_fusion([], [], k=60) == []
    assert [cid for cid, _ in reciprocal_rank_fusion([("a", 1.0)], [], k=60)] == ["a"]


# =============================================================================
# mmr_select() — pure logic, synthetic embeddings, no models
# =============================================================================

def _candidate(cid: str, relevance: float, embedding: List[float]) -> Dict[str, Any]:
    return {"complaint_id": cid, "relevance": relevance, "embedding": embedding}


def test_mmr_select_respects_top_n() -> None:
    candidates = [_candidate(str(i), float(i), [1.0, 0.0]) for i in range(10)]
    selected = mmr_select(candidates, top_n=3, lambda_=0.6)
    assert len(selected) == 3


def test_mmr_select_pure_relevance_when_lambda_is_one() -> None:
    """lambda=1 zeroes out the diversity term -> pure relevance ranking."""
    candidates = [
        _candidate("a", 0.9, [1.0, 0.0]),
        _candidate("b", 0.85, [1.0, 0.001]),  # near-duplicate of a
        _candidate("c", 0.5, [0.0, 1.0]),     # orthogonal / diverse
    ]
    selected = mmr_select(candidates, top_n=2, lambda_=1.0)
    assert selected == ["a", "b"], "with no diversity penalty, pure relevance order wins"


def test_mmr_select_prefers_diverse_over_near_duplicate() -> None:
    """
    The actual point of MMR: candidate 'b' is a near-duplicate of the
    already-selected 'a' and should lose out to the more diverse 'c',
    despite 'c' having lower raw relevance.
    """
    candidates = [
        _candidate("a", 0.9, [1.0, 0.0]),
        _candidate("b", 0.85, [1.0, 0.01]),  # nearly identical to a
        _candidate("c", 0.5, [0.0, 1.0]),    # orthogonal to a
    ]
    selected = mmr_select(candidates, top_n=2, lambda_=0.5)
    assert selected == ["a", "c"], "diverse-but-lower-relevance should beat a near-duplicate"


def test_mmr_select_empty_candidates() -> None:
    assert mmr_select([], top_n=5, lambda_=0.6) == []


# =============================================================================
# cross_encoder_rerank() and HybridSearchEngine — real models, shared once
# =============================================================================

@pytest.fixture(scope="module")
def cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.RERANKER_MODEL_NAME)


def make_corpus_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"complaint_id": "1", "company": "JPMORGAN CHASE & CO.", "product": "Checking or savings account",
         "cleaned_narrative": "Someone stole money from my checking account without my permission.",
         "date_received": "2023-01-01", "has_monetary_relief": True, "word_count": 12},
        {"complaint_id": "2", "company": "CITIBANK, N.A.", "product": "Mortgage",
         "cleaned_narrative": "My mortgage servicer misapplied my monthly payment twice this year.",
         "date_received": "2023-02-01", "has_monetary_relief": False, "word_count": 11},
        {"complaint_id": "3", "company": "WELLS FARGO & COMPANY", "product": "Credit card",
         "cleaned_narrative": "I was charged an annual fee on a credit card I was told had no fees.",
         "date_received": "2023-03-01", "has_monetary_relief": True, "word_count": 14},
    ])


def test_cross_encoder_rerank_ranks_relevant_pair_higher(cross_encoder: CrossEncoder) -> None:
    corpus = {row["complaint_id"]: row["cleaned_narrative"] for _, row in make_corpus_df().iterrows()}
    scores = cross_encoder_rerank(
        "unauthorized transaction on my account", ["1", "2", "3"], corpus, cross_encoder
    )
    assert scores["1"] > scores["2"]
    assert scores["1"] > scores["3"]


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory, cross_encoder: CrossEncoder) -> HybridSearchEngine:
    df = make_corpus_df()

    sparse = SparseIndex()
    sparse.build(df[["complaint_id", "cleaned_narrative"]].to_dict("records"))

    dense = DenseIndex(
        persist_dir=tmp_path_factory.mktemp("hybrid_chroma"),
        collection_name="hybrid_test",
        model_name=settings.EMBEDDING_MODEL_NAME,
    )
    dense.build(df)

    return HybridSearchEngine(
        sparse_index=sparse, dense_index=dense, corpus_df=df,
        reranker_model_name=settings.RERANKER_MODEL_NAME,
    )


def test_engine_search_returns_expected_shape(engine: HybridSearchEngine) -> None:
    results = engine.search("unauthorized transaction on my account", top_n_final=2)

    assert results
    for r in results:
        assert set(r.keys()) == {"complaint_id", "company", "product", "snippet", "relevance_score"}


def test_engine_search_finds_relevant_complaint_first(engine: HybridSearchEngine) -> None:
    results = engine.search("credit card annual fee I was told did not exist", top_n_final=1)
    assert results[0]["complaint_id"] == "3"
