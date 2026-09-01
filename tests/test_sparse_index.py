"""
Unit tests for the BM25 sparse (keyword) search index.
"""

from typing import Dict, List
import pandas as pd
import pytest

from src.config import settings
from src.retrieval.sparse_index import STOPWORDS, SparseIndex, tokenize


# =============================================================================
# tokenize()
# =============================================================================

def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Chase Refused My Refund!") == ["chase", "refused", "refund"]


def test_tokenize_drops_stopwords() -> None:
    tokens = tokenize("the bank did not refund my money")
    assert "the" not in tokens
    assert "did" not in tokens
    assert "bank" in tokens
    assert "refund" in tokens


def test_tokenize_preserves_regulation_citations() -> None:
    """'§1005.11' must survive as one token, not be split or lose the '§'."""
    tokens = tokenize("This violates Regulation E §1005.11 clearly.")
    assert "§1005.11" in tokens


def test_tokenize_handles_empty_and_none() -> None:
    assert tokenize("") == []
    assert tokenize(None) == []


def test_stopwords_do_not_include_domain_terms() -> None:
    """Sanity guard: a generic stopword list must not eat financial terms."""
    for term in ("chase", "mortgage", "refund", "unauthorized", "dispute"):
        assert term not in STOPWORDS


# =============================================================================
# SparseIndex
# =============================================================================

def make_records() -> List[Dict]:
    return [
        {"complaint_id": "1", "cleaned_narrative": "Chase refused to refund an unauthorized wire transfer from my checking account."},
        {"complaint_id": "2", "cleaned_narrative": "Wells Fargo charged an overdraft fee I never agreed to on my savings account."},
        {"complaint_id": "3", "cleaned_narrative": "My mortgage servicer at Citibank misapplied my monthly payment twice this year."},
    ]


def test_search_before_build_raises() -> None:
    index = SparseIndex()
    with pytest.raises(RuntimeError, match="build\\(\\) must be called"):
        index.search("chase wire transfer", top_k=5)


def test_search_ranks_matching_document_first() -> None:
    index = SparseIndex()
    index.build(make_records())

    results = index.search("unauthorized wire transfer chase", top_k=3)

    assert results, "expected at least one match"
    assert results[0][0] == "1"


def test_search_respects_top_k() -> None:
    index = SparseIndex()
    index.build(make_records())

    results = index.search("account", top_k=1)

    assert len(results) == 1


def test_search_excludes_zero_score_documents() -> None:
    index = SparseIndex()
    index.build(make_records())

    # Shares no meaningful terms with any of the three narratives.
    results = index.search("xyzxyz nonexistent gibberish term", top_k=3)

    assert results == []


def test_search_returns_complaint_ids_not_positions() -> None:
    """Regression guard: results must map back to real complaint_ids, not list indices."""
    records = list(reversed(make_records()))  # complaint_id "3" is now at position 0
    index = SparseIndex()
    index.build(records)

    results = index.search("wire transfer chase unauthorized", top_k=3)

    assert results[0][0] == "1", "must resolve to the real complaint_id, not its list position"


# =============================================================================
# Integration: real corpus (proves the "sub-second, no persistence needed" claim)
# =============================================================================

def test_builds_and_searches_real_corpus_fast() -> None:
    if not settings.PROCESSED_PARQUET_FILE.exists():
        pytest.skip("processed parquet not present in this environment")

    df = pd.read_parquet(settings.PROCESSED_PARQUET_FILE)
    records = df[["complaint_id", "cleaned_narrative"]].to_dict("records")

    index = SparseIndex()
    index.build(records)
    results = index.search("unauthorized wire transfer chase", top_k=10)

    assert len(results) > 0
    assert all(isinstance(cid, str) and score > 0 for cid, score in results)
