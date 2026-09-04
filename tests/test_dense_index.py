"""
Unit tests for the ChromaDB dense (meaning-based) search index.

The embedding model loads once per test module (~7s) via a module-scoped
fixture rather than once per test — reloading it per test would make this
file the slowest thing in the suite for no reason.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.config import settings
from src.retrieval.dense_index import DenseIndex, _to_chroma_scalar


def make_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "complaint_id": "1",
            "cleaned_narrative": "Someone stole money from my checking account without my permission.",
            "company": "JPMORGAN CHASE & CO.", "product": "Checking or savings account",
            "product_family": "Checking or savings account", "date_received": "2023-01-01",
            "has_monetary_relief": True, "word_count": 12,
        },
        {
            "complaint_id": "2",
            "cleaned_narrative": "My mortgage servicer misapplied my monthly payment twice this year.",
            "company": "CITIBANK, N.A.", "product": "Mortgage",
            "product_family": "Mortgage", "date_received": "2023-02-01",
            "has_monetary_relief": False, "word_count": 11,
        },
    ])


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> DenseIndex:
    persist_dir = tmp_path_factory.mktemp("chroma_test")
    idx = DenseIndex(
        persist_dir=persist_dir,
        collection_name="test_collection",
        model_name=settings.EMBEDDING_MODEL_NAME,
    )
    idx.build(make_df())
    return idx


# =============================================================================
# _to_chroma_scalar() — the metadata type conversion
# =============================================================================

def test_converts_datetime_date() -> None:
    """The real corpus's date_received comes back as datetime.date, not pd.Timestamp."""
    import datetime
    result = _to_chroma_scalar(datetime.date(2023, 12, 31))
    assert result == "2023-12-31"
    assert isinstance(result, str)


def test_converts_pandas_timestamp() -> None:
    result = _to_chroma_scalar(pd.Timestamp("2023-01-01"))
    assert result == "2023-01-01T00:00:00"


def test_converts_numpy_scalars() -> None:
    import numpy as np
    assert _to_chroma_scalar(np.int64(5)) == 5 and isinstance(_to_chroma_scalar(np.int64(5)), int)
    assert _to_chroma_scalar(np.bool_(True)) is True
    assert _to_chroma_scalar(np.float64(1.5)) == 1.5


def test_passes_through_plain_types() -> None:
    assert _to_chroma_scalar("hello") == "hello"
    assert _to_chroma_scalar(5) == 5
    assert _to_chroma_scalar(None) is None


# =============================================================================
# DenseIndex — semantic search behavior
# =============================================================================

def test_search_matches_by_meaning_not_just_keywords(index: DenseIndex) -> None:
    """
    The whole point of dense search: 'unauthorized transaction' should find
    complaint #1 ('someone stole money... without my permission') even
    though they share almost no exact words. BM25 alone would not do this.
    """
    results = index.search("unauthorized transaction on my account", top_k=2)

    assert results, "expected at least one match"
    assert results[0][0] == "1"


def test_search_returns_scores_and_embeddings(index: DenseIndex) -> None:
    results = index.search("mortgage payment error", top_k=2)

    for complaint_id, score, embedding in results:
        assert isinstance(complaint_id, str)
        assert 0.0 <= score <= 1.0, f"score {score} outside expected [0,1] range"
        assert len(embedding) == 384, "bge-small-en-v1.5 produces 384-dim vectors"
        assert all(isinstance(x, float) for x in embedding)


def test_build_is_idempotent(index: DenseIndex, mocker: Any) -> None:
    """A second build() with the same row count must skip re-embedding entirely."""
    spy = mocker.spy(index._model, "encode")
    index.build(make_df())
    spy.assert_not_called()


def test_build_chunks_upserts_without_dropping_rows(tmp_path: Path) -> None:
    """
    Regression guard: Chroma rejects a single upsert() above ~5461 records
    ("Batch size of 8831 is greater than max batch size of 5461") — hit for
    real building the full 8,831-row corpus, after 19 minutes of embedding
    work, because build() upserted everything in one call. Forces a tiny
    chunk size here so 5 rows require 3 separate upsert() calls, and checks
    every row survives split across them, none dropped or overwritten.
    """
    df = pd.DataFrame([
        {"complaint_id": str(i), "company": "JPMORGAN CHASE & CO.", "product": "Mortgage",
         "cleaned_narrative": f"Test narrative number {i} about a mortgage issue.",
         "date_received": "2023-01-01", "has_monetary_relief": False, "word_count": 8}
        for i in range(5)
    ])

    idx = DenseIndex(
        persist_dir=tmp_path, collection_name="chunk_test",
        model_name=settings.EMBEDDING_MODEL_NAME,
    )
    idx._UPSERT_CHUNK_SIZE = 2  # forces 3 chunks: [0,1], [2,3], [4]
    idx.build(df)

    assert idx._collection.count() == 5
    results = idx.search("mortgage issue", top_k=5)
    assert {cid for cid, _score, _emb in results} == {"0", "1", "2", "3", "4"}


# =============================================================================
# Integration: a slice of the real corpus (proves real dtypes don't crash build)
# =============================================================================

def test_builds_against_real_corpus_dtypes(tmp_path: Path) -> None:
    """
    Regression guard for the datetime.date bug found while building this:
    the real parquet's date_received is datetime.date, has_monetary_relief
    is np.bool_, word_count is np.int64 — none of these are plain Python
    types, and Chroma rejects all three if unconverted. Uses a 20-row slice,
    not the full 8,831, since embedding is the expensive step and this test
    only needs to prove the real column types don't crash build().
    """
    if not settings.PROCESSED_PARQUET_FILE.exists():
        pytest.skip("processed parquet not present in this environment")

    df = pd.read_parquet(settings.PROCESSED_PARQUET_FILE).head(20)
    idx = DenseIndex(
        persist_dir=tmp_path, collection_name="real_slice",
        model_name=settings.EMBEDDING_MODEL_NAME,
    )
    idx.build(df)  # must not raise

    results = idx.search("unauthorized charge on my account", top_k=5)
    assert len(results) > 0
