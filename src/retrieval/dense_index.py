"""
Dense (Meaning-Based) Search Index using ChromaDB + BGE Embeddings.

Embeds complaint narratives into vectors capturing semantic meaning, so a
query and a differently-worded complaint about the same problem can match
even without shared keywords. Complements sparse_index.py's exact keyword
search; see hybrid_search.py for how the two are fused.
"""

import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import settings

# BAAI/bge-* embedding models are trained "asymmetrically": a query needs this
# instruction prefix to retrieve well against passages, which are embedded
# as-is with no prefix. Omitting it on queries silently degrades retrieval
# quality rather than erroring, so it's easy to miss if you don't know to
# look for it in the model's own documentation.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Stored as Chroma metadata alongside each embedding so Phase 5's self-query
# filtering (e.g. "Chase mortgage complaints in Q3 2023") can filter on these
# without re-embedding the corpus later — embedding is the expensive step.
_METADATA_FIELDS = [
    "company", "product", "product_family", "date_received",
    "has_monetary_relief", "word_count",
]


class DenseIndex:
    """
    ChromaDB-backed dense vector index over complaint narratives.

    Persisted to disk at `persist_dir`. `build()` is idempotent: it skips
    re-embedding if the collection already holds the expected row count,
    since embedding thousands of narratives is the one genuinely expensive
    step in this pipeline and shouldn't be repeated on every run.
    """

    def __init__(self, persist_dir: Path, collection_name: str, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    # Chroma rejects a single upsert() call above this size (confirmed
    # directly: 8,831 records in one call raised "Batch size of 8831 is
    # greater than max batch size of 5461"). Kept comfortably under that.
    _UPSERT_CHUNK_SIZE = 2000

    def build(self, df: pd.DataFrame, batch_size: int = 64) -> None:
        """
        Embeds `cleaned_narrative` for every row and upserts into Chroma with
        `_METADATA_FIELDS` attached, in chunks of `_UPSERT_CHUNK_SIZE` rows.
        No-ops if the collection is already populated with `len(df)` records.

        Chunked rather than one encode-everything-then-upsert-everything
        call: each chunk is saved to disk as soon as it's embedded, so a
        crash or interruption partway through only loses the current chunk's
        work (minutes), not the entire run (previously ~19 minutes lost to
        the batch-size error above, in full).
        """
        if self._collection.count() == len(df):
            return

        # product_family is a convenience grouping (e.g. "Credit card" covers
        # several raw CFPB product spellings) — it was never persisted to the
        # parquet itself, only computed transiently for a CLI summary. Derive
        # it here if the caller didn't already provide it, rather than
        # silently requiring callers to remember an extra step.
        if "product_family" not in df.columns:
            df = df.copy()
            df["product_family"] = df["product"].map(settings.PRODUCT_TO_FAMILY).fillna("Other")

        for start in range(0, len(df), self._UPSERT_CHUNK_SIZE):
            chunk = df.iloc[start : start + self._UPSERT_CHUNK_SIZE]

            ids = chunk["complaint_id"].astype(str).tolist()
            texts = chunk["cleaned_narrative"].tolist()
            embeddings = self._model.encode(
                texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
            )
            metadatas = [
                {field: _to_chroma_scalar(row[field]) for field in _METADATA_FIELDS}
                for _, row in chunk.iterrows()
            ]

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )

    def search(self, query: str, top_k: int) -> List[Tuple[str, float, List[float]]]:
        """
        Returns (complaint_id, similarity_score, embedding) triples, best
        match first. The embedding is included so MMR (in hybrid_search.py)
        can reuse it without a second, redundant encode call.
        """
        query_embedding = self._model.encode(
            _QUERY_INSTRUCTION + query, convert_to_numpy=True
        )
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["distances", "embeddings"],
        )

        # Chroma's default space is squared L2 distance (0 = identical,
        # unbounded above). Convert to a 0-1 similarity score, higher-is-
        # better, so this lines up with sparse_index.py's BM25 scores —
        # both feed reciprocal rank fusion next, in hybrid_search.py.
        ids = result["ids"][0]
        distances = result["distances"][0]
        embeddings = result["embeddings"][0]
        return [
            (cid, 1.0 / (1.0 + dist), [float(x) for x in emb])
            for cid, dist, emb in zip(ids, distances, embeddings)
        ]

    def get_embeddings(self, ids: List[str]) -> Dict[str, List[float]]:
        """
        Looks up already-stored embeddings for the given complaint_ids
        without re-encoding — used by hybrid_search.py for candidates that
        came from BM25 rather than dense search, so MMR still has an
        embedding for every candidate without a redundant encode call.

        Chroma's get() does NOT preserve the requested id order in its
        response — verified directly: a request for ['b','a'] came back
        ordered ['a','b']. Results here are zipped against the *returned*
        id list, never assumed to match the input order.
        """
        if not ids:
            return {}
        result = self._collection.get(ids=ids, include=["embeddings"])
        return {
            cid: [float(x) for x in emb]
            for cid, emb in zip(result["ids"], result["embeddings"])
        }


def _to_chroma_scalar(value: Any) -> Any:
    """
    Chroma metadata values must be a plain str/int/float/bool — pandas and
    numpy types (numpy.int64, numpy.bool_, datetime.date, ...) are rejected
    outright at upsert time, not silently coerced. Verified directly against
    both a synthetic numpy.int64 and the real corpus's actual column types:
    `date_received` comes back from `pd.read_parquet` as plain
    `datetime.date` (not `pd.Timestamp`, which a naive Timestamp-only check
    would have missed) — `pd.Timestamp` is itself a `datetime.date`
    subclass, so one isinstance check below covers both.
    """
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
