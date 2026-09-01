"""
Hybrid Search Engine: RRF Fusion, Cross-Encoder Reranking, MMR Diversity.

Combines sparse_index.py (exact keyword match) and dense_index.py (semantic
meaning match) into one ranked result set. Pipeline, in order:

    BM25 top-K  ---\\
                     +--> RRF fuse --> Cross-Encoder rerank --> MMR select final top-N
    Dense top-K ---/

MMR runs LAST, after the cross-encoder, not before. If diversity were
enforced before the cross-encoder's pure-relevance rerank, that rerank could
simply resurface near-duplicate complaints and undo the diversity fix —
whichever step runs last determines what the caller actually sees. See
.work/phase3_hybrid_retrieval_plan.md section 3 for the full reasoning.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.dense_index import DenseIndex
from src.retrieval.sparse_index import SparseIndex


def reciprocal_rank_fusion(
    dense_results: List[Tuple[str, float]],
    sparse_results: List[Tuple[str, float]],
    k: int = settings.RRF_K,
) -> List[Tuple[str, float]]:
    """
    Fuses two independently-ranked result lists into one, by rank rather
    than by raw score — BM25 scores and cosine-derived similarity scores
    live on totally different, incomparable scales, but "rank 1", "rank 2"
    means the same thing in both lists.

    score(doc) = sum, over every list doc appears in, of 1 / (k + rank)

    A document ranked highly in *either* list scores well; it doesn't need
    to win both. `k` dampens the impact of small rank differences near the
    top (the standard value, from the original RRF paper, is 60).
    """
    scores: Dict[str, float] = {}
    for rank, (complaint_id, _score) in enumerate(dense_results, start=1):
        scores[complaint_id] = scores.get(complaint_id, 0.0) + 1.0 / (k + rank)
    for rank, (complaint_id, _score) in enumerate(sparse_results, start=1):
        scores[complaint_id] = scores.get(complaint_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def cross_encoder_rerank(
    query: str,
    candidate_ids: List[str],
    corpus: Dict[str, str],
    model: CrossEncoder,
) -> Dict[str, float]:
    """
    Scores each (query, narrative) pair with the cross-encoder — reading
    both together in one pass, unlike BM25/dense which score them
    independently. Far more accurate, but too slow to run on the whole
    corpus, hence only ever called on the ~50 RRF-fused candidates.

    Returns raw scores keyed by complaint_id, un-normalized — mmr_select()
    min-max normalizes them internally before combining with similarity.
    """
    pairs = [(query, corpus[cid]) for cid in candidate_ids]
    raw_scores = model.predict(pairs)
    return {cid: float(score) for cid, score in zip(candidate_ids, raw_scores)}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def mmr_select(
    candidates: List[Dict[str, Any]],
    top_n: int = settings.FINAL_TOP_N,
    lambda_: float = settings.MMR_LAMBDA,
) -> List[str]:
    """
    Greedy Maximal Marginal Relevance selection — picks the final top_n by
    trading cross-encoder relevance off against similarity to what's
    already been picked, so the result isn't five near-duplicate complaints.

        MMR(d) = lambda * relevance(d) - (1 - lambda) * max_sim(d, selected)

    `candidates`: each dict needs 'complaint_id', 'relevance' (raw
    cross-encoder score), and 'embedding' (dense vector, reused from
    retrieval — never re-encoded here). Relevance is min-max normalized to
    [0, 1] across the candidate set first, since raw cross-encoder scores
    and cosine similarity live on different scales.

    No query embedding is used here: 'relevance' already captures how well
    a candidate matches the query (via the cross-encoder), and the
    diversity term only needs similarity *between candidates* — adding the
    query vector on top would be redundant, not more accurate.
    """
    if not candidates:
        return []

    raw = [c["relevance"] for c in candidates]
    lo, hi = min(raw), max(raw)
    span = hi - lo
    normalized = {
        c["complaint_id"]: (c["relevance"] - lo) / span if span > 0 else 1.0
        for c in candidates
    }

    remaining = list(candidates)
    selected: List[Dict[str, Any]] = []

    while remaining and len(selected) < top_n:
        best_candidate, best_score = None, -float("inf")
        for candidate in remaining:
            if selected:
                max_sim = max(
                    _cosine_similarity(candidate["embedding"], s["embedding"])
                    for s in selected
                )
            else:
                max_sim = 0.0
            score = lambda_ * normalized[candidate["complaint_id"]] - (1 - lambda_) * max_sim
            if score > best_score:
                best_candidate, best_score = candidate, score

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return [c["complaint_id"] for c in selected]


class HybridSearchEngine:
    """Orchestrates the full pipeline: BM25 + dense -> RRF -> cross-encoder -> MMR."""

    def __init__(
        self,
        sparse_index: SparseIndex,
        dense_index: DenseIndex,
        corpus_df: pd.DataFrame,
        reranker_model_name: str = settings.RERANKER_MODEL_NAME,
    ) -> None:
        self._sparse = sparse_index
        self._dense = dense_index
        self._corpus = corpus_df.set_index("complaint_id")
        self._cross_encoder = CrossEncoder(reranker_model_name)

    def search(
        self,
        query: str,
        top_k_retrieve: int = settings.RETRIEVAL_TOP_K,
        rrf_k: int = settings.RRF_K,
        top_n_final: int = settings.FINAL_TOP_N,
        mmr_lambda: float = settings.MMR_LAMBDA,
    ) -> List[Dict[str, Any]]:
        """
        Returns up to top_n_final results as
        {complaint_id, company, product, snippet, relevance_score},
        best match first.
        """
        dense_hits = self._dense.search(query, top_k_retrieve)   # (id, score, embedding)
        sparse_hits = self._sparse.search(query, top_k_retrieve)  # (id, score)

        dense_ranked = [(cid, score) for cid, score, _emb in dense_hits]
        fused = reciprocal_rank_fusion(dense_ranked, sparse_hits, k=rrf_k)
        fused_ids = [cid for cid, _score in fused][:top_k_retrieve]

        if not fused_ids:
            return []

        # Reuse embeddings already fetched during dense retrieval; only look
        # up the ones that came from BM25-only hits.
        embedding_lookup = {cid: emb for cid, _score, emb in dense_hits}
        missing_ids = [cid for cid in fused_ids if cid not in embedding_lookup]
        if missing_ids:
            embedding_lookup.update(self._dense.get_embeddings(missing_ids))

        corpus_texts = {
            cid: self._corpus.loc[cid, "cleaned_narrative"] for cid in fused_ids
        }
        relevance = cross_encoder_rerank(query, fused_ids, corpus_texts, self._cross_encoder)

        candidates = [
            {"complaint_id": cid, "relevance": relevance[cid], "embedding": embedding_lookup[cid]}
            for cid in fused_ids
        ]
        selected_ids = mmr_select(candidates, top_n=top_n_final, lambda_=mmr_lambda)

        results = []
        for cid in selected_ids:
            row = self._corpus.loc[cid]
            narrative = row["cleaned_narrative"]
            snippet = narrative[:240] + ("..." if len(narrative) > 240 else "")
            results.append({
                "complaint_id": cid,
                "company": row["company"],
                "product": row["product"],
                "snippet": snippet,
                "relevance_score": relevance[cid],
            })
        return results
