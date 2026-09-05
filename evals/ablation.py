"""
Phase 6: Retrieval Ablation Study Benchmark Runner.

Scientifically quantifies the marginal value of each component in the retrieval pipeline:
1. BM25 Only (Lexical Baseline)
2. Dense BGE Only (Vector Semantic Baseline)
3. Hybrid + RRF (Reciprocal Rank Fusion)
4. Hybrid + RRF + Cross-Encoder Reranker
5. Full Pipeline (+ MMR Diversity Suppression)

Evaluates over the 45 answerable benchmark queries from Phase 4 using:
- Recall@50
- MRR@10 (Mean Reciprocal Rank)
- nDCG@10 (Normalized Discounted Cumulative Gain with graded relevance)
- p50 and p95 Latency (ms)
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid_search import (
    cross_encoder_rerank,
    mmr_select,
    reciprocal_rank_fusion,
)
from src.retrieval.sparse_index import SparseIndex
from evals.golden_set import get_query_partitions, get_relevance_map
from evals.metrics import mean_score, ndcg_at_k, recall_at_k, reciprocal_rank
from evals.queries import EVAL_QUERIES, EvalQuery

OUTPUT_JSON = settings.PROCESSED_DATA_DIR / "ablation_results.json"
OUTPUT_MARKDOWN = settings.PROCESSED_DATA_DIR / "ablation_results.md"
EVALS_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class AblationConfigResult:
    """Benchmark performance summary for a single retrieval configuration."""
    config_name: str
    description: str
    mrr_at_10: float
    ndcg_at_10: float
    recall_at_50: float
    p50_latency_ms: float
    p95_latency_ms: float


class AblationRunner:
    """Orchestrates multi-stage retrieval evaluation across benchmark queries."""

    def __init__(self) -> None:
        console = Console()
        console.print("[bold cyan][*] Loading corpus and initializing search engines...[/bold cyan]")

        self.df = pd.read_parquet(settings.PROCESSED_PARQUET_FILE)
        self.corpus_lookup = self.df.set_index("complaint_id")

        # 1. Sparse BM25 Index
        self.sparse = SparseIndex()
        self.sparse.build(self.df[["complaint_id", "cleaned_narrative"]].to_dict("records"))

        # 2. Dense ChromaDB Index
        self.dense = DenseIndex(
            persist_dir=settings.CHROMA_DB_DIR,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            model_name=settings.EMBEDDING_MODEL_NAME,
        )
        self.dense.build(self.df)

        # 3. Cross-Encoder Reranker
        self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)

        # 4. Load Ground Truth Relevance Data
        partitions = get_query_partitions()
        self.answerable_qids = set(partitions["answerable"])

        self.graded_map = get_relevance_map(graded=True)
        self.binary_map = get_relevance_map(graded=False)
        self.binary_positives = {
            qid: {cid for cid, s in docs.items() if s > 0}
            for qid, docs in self.binary_map.items()
        }

    def run_all(self, limit: Optional[int] = None) -> List[AblationConfigResult]:
        """
        Runs all 5 configurations across the answerable benchmark queries.
        """
        console = Console()
        eval_subset = [q for q in EVAL_QUERIES if q["query_id"] in self.answerable_qids]
        if limit:
            eval_subset = eval_subset[:limit]

        total_queries = len(eval_subset)
        console.print(f"[bold green][*] Running ablation benchmark across {total_queries} queries...[/bold green]\n")

        # Metric accumulators for each configuration
        configs = [
            ("BM25 Only", "Lexical BM25 keyword search baseline"),
            ("Dense BGE Only", "Semantic BGE-small dense vector embeddings"),
            ("Hybrid + RRF", "Reciprocal Rank Fusion (BM25 + Dense, k=60)"),
            ("Hybrid + Cross-Encoder", "Hybrid fused top-50 reranked via BGE Cross-Encoder"),
            ("Full Pipeline (+ MMR)", "Hybrid + Reranking + MMR diversity suppression (lambda=0.6)"),
        ]

        metrics_store: Dict[str, Dict[str, List[float]]] = {
            name: {"rr": [], "ndcg": [], "recall": [], "latencies": []}
            for name, _ in configs
        }

        for i, q in enumerate(eval_subset, start=1):
            qid = q["query_id"]
            query_text = q["query"]
            positives = self.binary_positives.get(qid, set())
            graded_truth = self.graded_map.get(qid, {})

            # -------------------------------------------------------------
            # Config 1: BM25 Only
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            sparse_hits = self.sparse.search(query_text, top_k=50)
            bm25_ids = [cid for cid, _score in sparse_hits]
            bm25_lat = (time.perf_counter() - t0) * 1000

            metrics_store["BM25 Only"]["rr"].append(reciprocal_rank(bm25_ids, positives, k=10))
            metrics_store["BM25 Only"]["ndcg"].append(ndcg_at_k(bm25_ids, graded_truth, k=10))
            metrics_store["BM25 Only"]["recall"].append(recall_at_k(bm25_ids, positives, k=50))
            metrics_store["BM25 Only"]["latencies"].append(bm25_lat)

            # -------------------------------------------------------------
            # Config 2: Dense BGE Only
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            dense_hits = self.dense.search(query_text, top_k=50)
            dense_ids = [cid for cid, _score, _emb in dense_hits]
            dense_lat = (time.perf_counter() - t0) * 1000

            metrics_store["Dense BGE Only"]["rr"].append(reciprocal_rank(dense_ids, positives, k=10))
            metrics_store["Dense BGE Only"]["ndcg"].append(ndcg_at_k(dense_ids, graded_truth, k=10))
            metrics_store["Dense BGE Only"]["recall"].append(recall_at_k(dense_ids, positives, k=50))
            metrics_store["Dense BGE Only"]["latencies"].append(dense_lat)

            # -------------------------------------------------------------
            # Config 3: Hybrid + RRF
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            dense_ranked = [(cid, score) for cid, score, _emb in dense_hits]
            fused = reciprocal_rank_fusion(dense_ranked, sparse_hits, k=settings.RRF_K)
            fused_ids = [cid for cid, _score in fused][:50]
            hybrid_lat = (time.perf_counter() - t0) * 1000 + bm25_lat + dense_lat

            metrics_store["Hybrid + RRF"]["rr"].append(reciprocal_rank(fused_ids, positives, k=10))
            metrics_store["Hybrid + RRF"]["ndcg"].append(ndcg_at_k(fused_ids, graded_truth, k=10))
            metrics_store["Hybrid + RRF"]["recall"].append(recall_at_k(fused_ids, positives, k=50))
            metrics_store["Hybrid + RRF"]["latencies"].append(hybrid_lat)

            # -------------------------------------------------------------
            # Config 4: Hybrid + Cross-Encoder Reranker
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            corpus_texts = {cid: self.corpus_lookup.loc[cid, "cleaned_narrative"] for cid in fused_ids}
            relevance_scores = cross_encoder_rerank(query_text, fused_ids, corpus_texts, self.reranker)
            reranked_ids = sorted(fused_ids, key=lambda cid: relevance_scores[cid], reverse=True)
            rerank_lat = (time.perf_counter() - t0) * 1000 + hybrid_lat

            metrics_store["Hybrid + Cross-Encoder"]["rr"].append(reciprocal_rank(reranked_ids, positives, k=10))
            metrics_store["Hybrid + Cross-Encoder"]["ndcg"].append(ndcg_at_k(reranked_ids, graded_truth, k=10))
            metrics_store["Hybrid + Cross-Encoder"]["recall"].append(recall_at_k(reranked_ids, positives, k=50))
            metrics_store["Hybrid + Cross-Encoder"]["latencies"].append(rerank_lat)

            # -------------------------------------------------------------
            # Config 5: Full Pipeline (+ MMR Diversity)
            # -------------------------------------------------------------
            t0 = time.perf_counter()
            embedding_lookup = {cid: emb for cid, _score, emb in dense_hits}
            missing_ids = [cid for cid in fused_ids if cid not in embedding_lookup]
            if missing_ids:
                embedding_lookup.update(self.dense.get_embeddings(missing_ids))

            candidates = [
                {"complaint_id": cid, "relevance": relevance_scores[cid], "embedding": embedding_lookup[cid]}
                for cid in fused_ids
            ]
            # Select top 10 with MMR for ranking metric evaluation
            mmr_ids = mmr_select(candidates, top_n=10, lambda_=settings.MMR_LAMBDA)
            mmr_lat = (time.perf_counter() - t0) * 1000 + rerank_lat

            metrics_store["Full Pipeline (+ MMR)"]["rr"].append(reciprocal_rank(mmr_ids, positives, k=10))
            metrics_store["Full Pipeline (+ MMR)"]["ndcg"].append(ndcg_at_k(mmr_ids, graded_truth, k=10))
            metrics_store["Full Pipeline (+ MMR)"]["recall"].append(recall_at_k(reranked_ids, positives, k=50))
            metrics_store["Full Pipeline (+ MMR)"]["latencies"].append(mmr_lat)

            console.print(f"  [{i:2d}/{total_queries}] Processed {qid}: {query_text[:50]}...")

        # Aggregate Results
        summary_results: List[AblationConfigResult] = []
        for name, desc in configs:
            m = metrics_store[name]
            lats = m["latencies"]
            summary_results.append(
                AblationConfigResult(
                    config_name=name,
                    description=desc,
                    mrr_at_10=round(mean_score(m["rr"]), 4),
                    ndcg_at_10=round(mean_score(m["ndcg"]), 4),
                    recall_at_50=round(mean_score(m["recall"]), 4),
                    p50_latency_ms=round(float(np.percentile(lats, 50)), 2),
                    p95_latency_ms=round(float(np.percentile(lats, 95)), 2),
                )
            )

        return summary_results


def print_ablation_table(results: List[AblationConfigResult]) -> None:
    """Renders a rich comparative benchmark table in the terminal."""
    console = Console()
    baseline = results[0]  # BM25 baseline

    table = Table(
        title="[ABLATION] Phase 6: Retrieval Ablation Benchmark Study",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Configuration", style="bold")
    table.add_column("Recall@50", justify="right")
    table.add_column("MRR@10", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("MRR Lift vs BM25", justify="right")
    table.add_column("p50 Latency", justify="right")
    table.add_column("p95 Latency", justify="right")

    for r in results:
        mrr_lift = ((r.mrr_at_10 - baseline.mrr_at_10) / baseline.mrr_at_10) * 100 if baseline.mrr_at_10 > 0 else 0.0
        lift_str = f"[green]+{mrr_lift:.1f}%[/green]" if mrr_lift > 0 else ("0.0%" if mrr_lift == 0 else f"[red]{mrr_lift:.1f}%[/red]")

        table.add_row(
            r.config_name,
            f"{r.recall_at_50 * 100:.1f}%",
            f"{r.mrr_at_10:.4f}",
            f"{r.ndcg_at_10:.4f}",
            lift_str,
            f"{r.p50_latency_ms:.1f}ms",
            f"{r.p95_latency_ms:.1f}ms",
        )

    console.print("\n", table, "\n")


def export_reports(results: List[AblationConfigResult]) -> None:
    """Exports results to both JSON and publication-ready Markdown table."""
    EVALS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    for path in [OUTPUT_JSON, EVALS_RESULTS_DIR / "ablation_results.json"]:
        with path.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    # 2. Save Markdown
    baseline = results[0]
    md_lines = [
        "# Retrieval Ablation Study Benchmark Results",
        "",
        "Empirical performance comparison across 5 retrieval architectures on the CFPB Consumer Complaint Intelligence dataset.",
        "",
        "| Retrieval Configuration | Recall@50 | MRR@10 | nDCG@10 | MRR Lift vs BM25 | p50 Latency | p95 Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        lift = ((r.mrr_at_10 - baseline.mrr_at_10) / baseline.mrr_at_10) * 100 if baseline.mrr_at_10 > 0 else 0.0
        lift_str = f"+{lift:.1f}%" if lift > 0 else ("0.0%" if lift == 0 else f"{lift:.1f}%")
        md_lines.append(
            f"| **{r.config_name}** | {r.recall_at_50 * 100:.1f}% | {r.mrr_at_10:.4f} | {r.ndcg_at_10:.4f} | {lift_str} | {r.p50_latency_ms:.1f}ms | {r.p95_latency_ms:.1f}ms |"
        )

    md_lines.append("")
    md_lines.append("> **Evaluation Protocol:** Measured across 45 answerable compliance queries from the frozen `golden_set.jsonl` benchmark (TREC-style pooled judging across 1,824 candidate dockets).")

    for path in [OUTPUT_MARKDOWN, EVALS_RESULTS_DIR / "ablation_results.md"]:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Retrieval Ablation Benchmark Study")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for smoke testing")
    args = parser.parse_args()

    runner = AblationRunner()
    results = runner.run_all(limit=args.limit)

    print_ablation_table(results)
    export_reports(results)


if __name__ == "__main__":
    main()
