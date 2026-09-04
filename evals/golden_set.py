"""
Golden Evaluation Set Loader, Validator, and Relevance Mapping Interface.

Loads the compiled ground-truth judgements from evals/golden_set.jsonl,
validates schema and label completeness, and formats relevance scores for
IR benchmark metrics (MRR@10, nDCG@10, Recall@50 in Phase 6).

Standard TREC IR convention:
- 'Answerable Queries': Queries with >= 1 relevant document in the corpus.
  These are used for ranking metrics (MRR@10, Recall@50, nDCG@10).
- 'Unanswerable Queries': Real-world queries where the underlying institution
  does not offer the product (e.g. Capital One mortgages in a 2022-2023 corpus).
  Preserved in the benchmark to test negative precision and hallucination resistance.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from rich.console import Console
from rich.table import Table

from evals.queries import EVAL_QUERIES

GOLDEN_SET_FILE = Path(__file__).resolve().parent / "golden_set.jsonl"

LabelType = Literal["relevant", "partially_relevant", "not_relevant"]

# Numerical score mapping for graded relevance (used by nDCG)
GRADED_RELEVANCE_SCORES: Dict[str, int] = {
    "relevant": 2,
    "partially_relevant": 1,
    "not_relevant": 0,
}

# Binary score mapping (used by MRR and Recall)
# By standard convention, partially_relevant is treated as 0 for strict metrics
BINARY_RELEVANCE_SCORES: Dict[str, int] = {
    "relevant": 1,
    "partially_relevant": 0,
    "not_relevant": 0,
}


@dataclass(frozen=True)
class GoldenJudgement:
    """A single judged (query, complaint) pair."""
    query_id: str
    query: str
    target_company: Optional[str]
    target_product_family: Optional[str]
    complaint_id: str
    company: str
    product: str
    snippet: str
    found_by: List[str]
    label: LabelType
    reasoning: str


def load_golden_set(filepath: Optional[Path] = None) -> List[GoldenJudgement]:
    """
    Loads all judged pairs from golden_set.jsonl.

    Raises:
        FileNotFoundError: If the golden set file does not exist.
        ValueError: If a row contains invalid JSON or missing required fields.
    """
    target_path = filepath or GOLDEN_SET_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"Golden set file not found at: {target_path}")

    judgements: List[GoldenJudgement] = []
    with target_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupt JSON at line {line_num}: {e}") from e

            label = data.get("label")
            if label not in GRADED_RELEVANCE_SCORES:
                raise ValueError(
                    f"Line {line_num}: Invalid or missing label '{label}' for query {data.get('query_id')}"
                )

            judgements.append(
                GoldenJudgement(
                    query_id=data["query_id"],
                    query=data["query"],
                    target_company=data.get("target_company"),
                    target_product_family=data.get("target_product_family"),
                    complaint_id=str(data["complaint_id"]),
                    company=data["company"],
                    product=data["product"],
                    snippet=data["snippet"],
                    found_by=data.get("found_by", []),
                    label=label,
                    reasoning=data.get("reasoning", ""),
                )
            )

    return judgements


def get_relevance_map(
    graded: bool = True,
    filepath: Optional[Path] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Builds a nested dictionary of ground-truth relevance judgements:
        relevance_map[query_id][complaint_id] = score (int)

    Args:
        graded: If True, uses graded scores (2 = relevant, 1 = partially_relevant, 0 = not_relevant).
                If False, uses binary scores (1 = relevant, 0 = non-relevant).
    """
    score_map = GRADED_RELEVANCE_SCORES if graded else BINARY_RELEVANCE_SCORES
    judgements = load_golden_set(filepath)

    relevance_map: Dict[str, Dict[str, int]] = {}
    for j in judgements:
        if j.query_id not in relevance_map:
            relevance_map[j.query_id] = {}
        relevance_map[j.query_id][j.complaint_id] = score_map[j.label]

    return relevance_map


def get_query_partitions(filepath: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Partitions the 50 queries into answerable vs unanswerable subsets based on
    strict binary relevance.

    Returns:
        {
            "answerable": List of query_ids with >= 1 relevant complaint (used for MRR/Recall),
            "unanswerable": List of query_ids with 0 relevant complaints (used for negative tests)
        }
    """
    binary_map = get_relevance_map(graded=False, filepath=filepath)
    answerable = []
    unanswerable = []

    for q in EVAL_QUERIES:
        qid = q["query_id"]
        q_scores = binary_map.get(qid, {})
        has_positive = any(score > 0 for score in q_scores.values())
        if has_positive:
            answerable.append(qid)
        else:
            unanswerable.append(qid)

    return {
        "answerable": answerable,
        "unanswerable": unanswerable,
    }


def audit_golden_set(filepath: Optional[Path] = None) -> Dict[str, Any]:
    """
    Performs integrity and distribution checks on the golden evaluation set.

    Validates:
      1. All 50 queries defined in evals/queries.py are represented.
      2. No judgements have null or missing labels.
      3. Separates answerable benchmark queries from domain-empty negative queries.
    """
    judgements = load_golden_set(filepath)
    expected_qids = {q["query_id"] for q in EVAL_QUERIES}
    present_qids = {j.query_id for j in judgements}

    missing_qids = expected_qids - present_qids
    unexpected_qids = present_qids - expected_qids

    label_counts: Dict[str, int] = {"relevant": 0, "partially_relevant": 0, "not_relevant": 0}
    per_query_judgements: Dict[str, List[GoldenJudgement]] = {}

    for j in judgements:
        label_counts[j.label] = label_counts.get(j.label, 0) + 1
        per_query_judgements.setdefault(j.query_id, []).append(j)

    partitions = get_query_partitions(filepath)

    pool_sizes = [len(docs) for docs in per_query_judgements.values()]
    avg_pool_size = sum(pool_sizes) / max(len(pool_sizes), 1)

    return {
        "total_judgements": len(judgements),
        "total_queries": len(present_qids),
        "expected_queries": len(expected_qids),
        "missing_queries": sorted(missing_qids),
        "unexpected_queries": sorted(unexpected_qids),
        "label_counts": label_counts,
        "answerable_queries": partitions["answerable"],
        "unanswerable_queries": partitions["unanswerable"],
        "min_pool_size": min(pool_sizes) if pool_sizes else 0,
        "max_pool_size": max(pool_sizes) if pool_sizes else 0,
        "avg_pool_size": avg_pool_size,
    }


def print_audit_report() -> None:
    """Prints a formatted terminal audit report using Rich."""
    console = Console()
    stats = audit_golden_set()

    console.print("\n[bold cyan]📊 Golden Evaluation Set Audit Report[/bold cyan]")
    console.print(f"File: [yellow]{GOLDEN_SET_FILE}[/yellow]\n")

    # Overview Table
    overview_table = Table(title="Dataset Overview", show_header=True, header_style="bold magenta")
    overview_table.add_column("Metric", style="dim")
    overview_table.add_column("Value", justify="right")

    overview_table.add_row("Total Labeled Pairs", f"{stats['total_judgements']:,}")
    overview_table.add_row("Total Queries Labeled", f"{stats['total_queries']} / {stats['expected_queries']}")
    overview_table.add_row("  ↳ Answerable Queries (Benchmark Active)", f"{len(stats['answerable_queries'])}")
    overview_table.add_row("  ↳ Unanswerable Queries (Negative Controls)", f"{len(stats['unanswerable_queries'])}")
    overview_table.add_row("Average Candidates / Query", f"{stats['avg_pool_size']:.1f}")
    overview_table.add_row("Min / Max Pool Size", f"{stats['min_pool_size']} / {stats['max_pool_size']}")
    console.print(overview_table)

    # Label Distribution Table
    dist_table = Table(title="Label Distribution", show_header=True, header_style="bold blue")
    dist_table.add_column("Relevance Class", style="bold")
    dist_table.add_column("Count", justify="right")
    dist_table.add_column("Percentage", justify="right")

    total = max(stats["total_judgements"], 1)
    for label, count in stats["label_counts"].items():
        pct = (count / total) * 100
        color = "green" if label == "relevant" else ("yellow" if label == "partially_relevant" else "red")
        dist_table.add_row(f"[{color}]{label}[/{color}]", f"{count:,}", f"{pct:.1f}%")
    console.print(dist_table)

    # Integrity Health Checks
    console.print("\n[bold]Integrity Health Checks:[/bold]")
    if stats["missing_queries"]:
        console.print(f"[bold red]❌ Missing queries:[/bold red] {stats['missing_queries']}")
    else:
        console.print("[green]✅ All 50 expected queries are present.[/green]")

    console.print(f"[green]✅ {len(stats['answerable_queries'])} queries verified with ground-truth positives for MRR/Recall ablation.[/green]")
    if stats["unanswerable_queries"]:
        console.print(
            f"[yellow]ℹ️  {len(stats['unanswerable_queries'])} queries have 0 corpus matches (e.g. Capital One mortgages):[/yellow] "
            f"{stats['unanswerable_queries']}"
        )
        console.print("   [dim]These are retained as negative controls to measure precision against unanswerable questions.[/dim]")


if __name__ == "__main__":
    print_audit_report()
