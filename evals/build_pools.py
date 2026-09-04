"""
Phase 4: builds labeling pools for the golden evaluation set.

For each query in evals/queries.py, runs BM25-only and dense-only retrieval
SEPARATELY — never the fused/reranked hybrid pipeline — and unions their
top-K results into one shortlist per query. This is "pooling": the standard
TREC method for building relevance judgments. Labeling only what either
retriever already surfaced (instead of all 8,831 complaints) keeps labeling
tractable, and pooling both retrievers rather than just one avoids biasing
the golden set toward whichever retriever the final system leans on.

Output: evals/pools.jsonl — one row per (query, candidate) pair, with
"label": null. A separate labeling pass fills labels in; this script only
builds the shortlist.
"""

import json
from pathlib import Path

import pandas as pd

from src.config import settings
from src.retrieval.dense_index import DenseIndex
from src.retrieval.sparse_index import SparseIndex
from evals.queries import EVAL_QUERIES

POOL_TOP_K = 20
OUTPUT_FILE = Path(__file__).resolve().parent / "pools.jsonl"


def build_pools() -> None:
    df = pd.read_parquet(settings.PROCESSED_PARQUET_FILE)
    corpus_lookup = df.set_index("complaint_id")

    print(f"Building BM25 index over {len(df)} complaints...")
    sparse = SparseIndex()
    sparse.build(df[["complaint_id", "cleaned_narrative"]].to_dict("records"))

    print("Building/loading persistent dense index (first run embeds all "
          f"{len(df)} rows — this is the expensive, one-time step)...")
    dense = DenseIndex(
        persist_dir=settings.CHROMA_DB_DIR,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        model_name=settings.EMBEDDING_MODEL_NAME,
    )
    dense.build(df)

    rows = []
    for i, eq in enumerate(EVAL_QUERIES, start=1):
        sparse_hits = sparse.search(eq["query"], top_k=POOL_TOP_K)
        dense_hits = dense.search(eq["query"], top_k=POOL_TOP_K)

        found_by: dict = {}
        for cid, _score in sparse_hits:
            found_by.setdefault(cid, set()).add("bm25")
        for cid, _score, _emb in dense_hits:
            found_by.setdefault(cid, set()).add("dense")

        for cid, sources in found_by.items():
            row = corpus_lookup.loc[cid]
            rows.append({
                "query_id": eq["query_id"],
                "query": eq["query"],
                "target_company": eq["company"],
                "target_product_family": eq["product_family"],
                "complaint_id": cid,
                "company": row["company"],
                "product": row["product"],
                "snippet": row["cleaned_narrative"][:300],
                "found_by": sorted(sources),
                "label": None,
            })

        print(f"  [{i:2d}/{len(EVAL_QUERIES)}] {eq['query_id']}: "
              f"{len(found_by)} pooled candidates ({eq['query'][:60]}...)")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} candidate rows across {len(EVAL_QUERIES)} "
          f"queries to {OUTPUT_FILE}")
    print(f"Average pool size: {len(rows) / len(EVAL_QUERIES):.1f} candidates/query")


if __name__ == "__main__":
    build_pools()
