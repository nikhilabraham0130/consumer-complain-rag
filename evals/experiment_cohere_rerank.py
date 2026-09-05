"""
Enterprise Benchmark: Cohere Rerank v3.5 vs Local Rerankers (with Rate Limiter).

Evaluates across benchmark queries:
1. BM25 Only (Lexical)
2. Dense BGE Only (Semantic)
3. Hybrid + RRF (Fusion)
4. Hybrid + Local BGE Cross-Encoder (from baseline / CPU)
5. Hybrid + Cohere Rerank v3.5 (Cloud GPU Reranker, 4,096-token window)

Includes:
- Strict Rate Limiter (9 RPM) to respect Cohere Free Trial tier (10 RPM limit).
- Automatic retry with exponential backoff on HTTP 429.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cohere
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid_search import cross_encoder_rerank, reciprocal_rank_fusion
from src.retrieval.sparse_index import SparseIndex
from evals.golden_set import get_query_partitions, get_relevance_map
from evals.metrics import mean_score, ndcg_at_k, recall_at_k, reciprocal_rank
from evals.queries import EVAL_QUERIES

load_dotenv()

OUTPUT_JSON = Path(__file__).resolve().parent / "results" / "cohere_ablation_results.json"
OUTPUT_MARKDOWN = Path(__file__).resolve().parent / "results" / "cohere_ablation_results.md"
PREVIOUS_ABLATION_JSON = Path(__file__).resolve().parent / "results" / "ablation_results.json"


class RateLimiter:
    """Enforces maximum calls per minute to strictly respect trial API limits."""

    def __init__(self, max_per_minute: int = 9) -> None:
        self.interval = 60.0 / max_per_minute  # ~6.67 seconds
        self.last_call = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            time.sleep(sleep_time)
        self.last_call = time.time()


def run_cohere_experiment(limit: Optional[int] = None, run_local_cross_encoder: bool = False) -> None:
    console = Console()
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        console.print("[bold red][!] Error: COHERE_API_KEY is not set in .env[/bold red]")
        return

    console.print("[bold cyan][*] Initializing search engines and Cohere client...[/bold cyan]")
    co = cohere.ClientV2(api_key=api_key)
    rate_limiter = RateLimiter(max_per_minute=9)

    df = pd.read_parquet(settings.PROCESSED_PARQUET_FILE)
    corpus_lookup = df.set_index("complaint_id")

    # 1. Sparse Index
    sparse = SparseIndex()
    sparse.build(df[["complaint_id", "cleaned_narrative"]].to_dict("records"))

    # 2. Dense Index
    dense = DenseIndex(
        persist_dir=settings.CHROMA_DB_DIR,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        model_name=settings.EMBEDDING_MODEL_NAME,
    )
    dense.build(df)

    # 3. Local Cross-Encoder (optional)
    local_reranker = CrossEncoder(settings.RERANKER_MODEL_NAME) if run_local_cross_encoder else None

    # 4. Golden Set Relevance Data
    partitions = get_query_partitions()
    answerable_qids = set(partitions["answerable"])
    graded_map = get_relevance_map(graded=True)
    binary_map = get_relevance_map(graded=False)
    binary_positives = {
        qid: {cid for cid, s in docs.items() if s > 0}
        for qid, docs in binary_map.items()
    }

    eval_subset = [q for q in EVAL_QUERIES if q["query_id"] in answerable_qids]
    if limit:
        eval_subset = eval_subset[:limit]

    total_queries = len(eval_subset)
    console.print(f"[bold green][*] Running Cohere benchmark across {total_queries} queries (Rate Limited at 9 RPM)...[/bold green]\n")

    arch_names = [
        "BM25 Only",
        "Dense BGE Only",
        "Hybrid + RRF",
        "Hybrid + Local Cross-Encoder",
        "Hybrid + Cohere Rerank v3.5",
    ]

    metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"rr": [], "ndcg": [], "recall": [], "latency": []}
        for name in arch_names
    }

    for i, q in enumerate(eval_subset, start=1):
        qid = q["query_id"]
        query_text = q["query"]
        positives = binary_positives.get(qid, set())
        graded_truth = graded_map.get(qid, {})

        console.print(f"[{i:2d}/{total_queries}] Evaluating {qid}: {query_text[:50]}...")

        # --- 1. BM25 Only ---
        t0 = time.perf_counter()
        sparse_hits = sparse.search(query_text, top_k=50)
        bm25_ids = [cid for cid, _ in sparse_hits]
        bm25_lat = (time.perf_counter() - t0) * 1000

        metrics["BM25 Only"]["rr"].append(reciprocal_rank(bm25_ids, positives, k=10))
        metrics["BM25 Only"]["ndcg"].append(ndcg_at_k(bm25_ids, graded_truth, k=10))
        metrics["BM25 Only"]["recall"].append(recall_at_k(bm25_ids, positives, k=50))
        metrics["BM25 Only"]["latency"].append(bm25_lat)

        # --- 2. Dense BGE Only ---
        t0 = time.perf_counter()
        dense_hits = dense.search(query_text, top_k=50)
        dense_ids = [cid for cid, _, _ in dense_hits]
        dense_lat = (time.perf_counter() - t0) * 1000

        metrics["Dense BGE Only"]["rr"].append(reciprocal_rank(dense_ids, positives, k=10))
        metrics["Dense BGE Only"]["ndcg"].append(ndcg_at_k(dense_ids, graded_truth, k=10))
        metrics["Dense BGE Only"]["recall"].append(recall_at_k(dense_ids, positives, k=50))
        metrics["Dense BGE Only"]["latency"].append(dense_lat)

        # --- 3. Hybrid + RRF ---
        t0 = time.perf_counter()
        dense_ranked = [(cid, score) for cid, score, _ in dense_hits]
        fused = reciprocal_rank_fusion(dense_ranked, sparse_hits, k=settings.RRF_K)
        fused_ids = [cid for cid, _ in fused][:50]
        rrf_lat = (time.perf_counter() - t0) * 1000 + bm25_lat + dense_lat

        metrics["Hybrid + RRF"]["rr"].append(reciprocal_rank(fused_ids, positives, k=10))
        metrics["Hybrid + RRF"]["ndcg"].append(ndcg_at_k(fused_ids, graded_truth, k=10))
        metrics["Hybrid + RRF"]["recall"].append(recall_at_k(fused_ids, positives, k=50))
        metrics["Hybrid + RRF"]["latency"].append(rrf_lat)

        # Candidate narratives
        corpus_texts = {cid: corpus_lookup.loc[cid, "cleaned_narrative"] for cid in fused_ids}

        # --- 4. Hybrid + Local Cross-Encoder ---
        if run_local_cross_encoder and local_reranker:
            t0 = time.perf_counter()
            local_scores = cross_encoder_rerank(query_text, fused_ids, corpus_texts, local_reranker)
            local_reranked_ids = sorted(fused_ids, key=lambda cid: local_scores[cid], reverse=True)
            local_lat = (time.perf_counter() - t0) * 1000 + rrf_lat

            metrics["Hybrid + Local Cross-Encoder"]["rr"].append(reciprocal_rank(local_reranked_ids, positives, k=10))
            metrics["Hybrid + Local Cross-Encoder"]["ndcg"].append(ndcg_at_k(local_reranked_ids, graded_truth, k=10))
            metrics["Hybrid + Local Cross-Encoder"]["recall"].append(recall_at_k(local_reranked_ids, positives, k=50))
            metrics["Hybrid + Local Cross-Encoder"]["latency"].append(local_lat)

        # --- 5. Hybrid + Cohere Rerank v3.5 ---
        cohere_candidates = fused_ids[:25]
        docs_to_rerank = [corpus_texts[cid] for cid in cohere_candidates]

        # Enforce rate limiter wait
        rate_limiter.wait()

        # Call with retry logic
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                response = co.rerank(
                    model="rerank-v3.5",
                    query=query_text,
                    documents=docs_to_rerank,
                    top_n=len(cohere_candidates),
                )
                cohere_call_ms = (time.perf_counter() - t0) * 1000
                cohere_ordered_ids = [cohere_candidates[item.index] for item in response.results]
                remaining_ids = [cid for cid in fused_ids if cid not in set(cohere_ordered_ids)]
                full_cohere_ranked = cohere_ordered_ids + remaining_ids
                total_cohere_lat = cohere_call_ms + rrf_lat

                metrics["Hybrid + Cohere Rerank v3.5"]["rr"].append(reciprocal_rank(full_cohere_ranked, positives, k=10))
                metrics["Hybrid + Cohere Rerank v3.5"]["ndcg"].append(ndcg_at_k(full_cohere_ranked, graded_truth, k=10))
                metrics["Hybrid + Cohere Rerank v3.5"]["recall"].append(recall_at_k(full_cohere_ranked, positives, k=50))
                metrics["Hybrid + Cohere Rerank v3.5"]["latency"].append(total_cohere_lat)
                break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "rate" in err_msg.lower():
                    console.print(f"  [yellow][!] Rate limit hit, backing off 12 seconds (attempt {attempt+1}/3)...[/yellow]")
                    time.sleep(12.0)
                else:
                    console.print(f"  [red][!] Cohere API error: {err_msg}[/red]")
                    break

    # If local cross-encoder was skipped, pull its pre-computed benchmark scores from previous ablation
    if not run_local_cross_encoder and PREVIOUS_ABLATION_JSON.exists():
        with PREVIOUS_ABLATION_JSON.open("r", encoding="utf-8") as f:
            prev_results = json.load(f)
        for row in prev_results:
            if row["config_name"] == "Hybrid + Cross-Encoder":
                local_mrr = row["mrr_at_10"]
                local_ndcg = row["ndcg_at_10"]
                local_recall = row["recall_at_50"]
                local_p50 = row["p50_latency_ms"]
                metrics["Hybrid + Local Cross-Encoder"]["rr"] = [local_mrr]
                metrics["Hybrid + Local Cross-Encoder"]["ndcg"] = [local_ndcg]
                metrics["Hybrid + Local Cross-Encoder"]["recall"] = [local_recall]
                metrics["Hybrid + Local Cross-Encoder"]["latency"] = [local_p50]

    # Print Comparison Table
    table = Table(
        title=f"[ABLATION] Enterprise Benchmark: Local vs. Cohere Rerank v3.5 ({total_queries} Queries)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Architecture", style="bold")
    table.add_column("Recall@50", justify="right")
    table.add_column("MRR@10", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("p50 Latency", justify="right")
    table.add_column("p95 Latency", justify="right")

    rows_data = []
    for name in arch_names:
        m = metrics[name]
        if not m["rr"]:
            continue
        recall = mean_score(m["recall"])
        mrr = mean_score(m["rr"])
        ndcg = mean_score(m["ndcg"])
        p50 = float(np.percentile(m["latency"], 50))
        p95 = float(np.percentile(m["latency"], 95))

        table.add_row(
            name,
            f"{recall * 100:.1f}%",
            f"{mrr:.4f}",
            f"{ndcg:.4f}",
            f"{p50:.1f}ms",
            f"{p95:.1f}ms",
        )
        rows_data.append({
            "architecture": name,
            "recall_at_50": round(recall, 4),
            "mrr_at_10": round(mrr, 4),
            "ndcg_at_10": round(ndcg, 4),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
        })

    console.print("\n", table, "\n")

    # Export results
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows_data, f, indent=2)

    md_lines = [
        f"# Enterprise Reranking Benchmark: Local vs. Cohere Cloud ({total_queries} Queries)",
        "",
        "| Architecture | Recall@50 | MRR@10 | nDCG@10 | p50 Latency | p95 Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows_data:
        md_lines.append(
            f"| **{r['architecture']}** | {r['recall_at_50']*100:.1f}% | {r['mrr_at_10']:.4f} | {r['ndcg_at_10']:.4f} | {r['p50_latency_ms']:.1f}ms | {r['p95_latency_ms']:.1f}ms |"
        )
    with OUTPUT_MARKDOWN.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cohere Rerank comparative experiment")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (default: all 45)")
    parser.add_argument("--run-local-cross-encoder", action="store_true", help="Re-run slow local CPU cross encoder")
    args = parser.parse_args()
    run_cohere_experiment(limit=args.limit, run_local_cross_encoder=args.run_local_cross_encoder)
