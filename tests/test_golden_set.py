"""
Unit Tests for Phase 4 Golden Evaluation Dataset.
Verifies schema compliance, label validity, query coverage, and relevance mapping.
"""

from pathlib import Path
import pytest

from evals.golden_set import (
    GOLDEN_SET_FILE,
    GRADED_RELEVANCE_SCORES,
    audit_golden_set,
    get_query_partitions,
    get_relevance_map,
    load_golden_set,
)
from evals.queries import EVAL_QUERIES


def test_golden_set_file_exists() -> None:
    """Ensure golden_set.jsonl exists on disk and is non-empty."""
    assert GOLDEN_SET_FILE.exists(), f"Missing golden set file at {GOLDEN_SET_FILE}"
    assert GOLDEN_SET_FILE.stat().st_size > 0, "Golden set file is empty"


def test_load_golden_set_schema() -> None:
    """Verifies that all candidate judgements parse correctly into GoldenJudgement objects."""
    judgements = load_golden_set()
    assert len(judgements) >= 1800, f"Expected >= 1,800 judgements, found {len(judgements)}"

    for j in judgements:
        assert j.query_id.startswith("q"), f"Invalid query_id format: {j.query_id}"
        assert j.complaint_id, "Missing complaint_id"
        assert j.company, f"Missing company for {j.complaint_id}"
        assert j.product, f"Missing product for {j.complaint_id}"
        assert len(j.snippet) > 0, f"Empty snippet for {j.complaint_id}"
        assert j.label in GRADED_RELEVANCE_SCORES, f"Unknown label: {j.label}"
        assert len(j.reasoning) > 0, f"Empty reasoning for {j.complaint_id}"


def test_all_expected_queries_covered() -> None:
    """Verifies that all 50 queries defined in evals/queries.py exist in the golden set."""
    judgements = load_golden_set()
    present_qids = {j.query_id for j in judgements}
    expected_qids = {q["query_id"] for q in EVAL_QUERIES}

    assert present_qids == expected_qids, (
        f"Mismatch in query IDs. Missing: {expected_qids - present_qids}, "
        f"Unexpected: {present_qids - expected_qids}"
    )


def test_relevance_map_graded_and_binary() -> None:
    """Verifies score translation for both graded (nDCG) and binary (MRR/Recall) metrics."""
    graded_map = get_relevance_map(graded=True)
    binary_map = get_relevance_map(graded=False)

    assert len(graded_map) == 50
    assert len(binary_map) == 50

    # Test score bounds
    for qid, docs in graded_map.items():
        assert len(docs) >= 20, f"Query {qid} has suspiciously small pool size ({len(docs)})"
        for cid, score in docs.items():
            assert score in (0, 1, 2), f"Invalid graded score {score} for {qid}:{cid}"

    for qid, docs in binary_map.items():
        for cid, score in docs.items():
            assert score in (0, 1), f"Invalid binary score {score} for {qid}:{cid}"


def test_query_partitions_and_answerability() -> None:
    """
    Verifies that the answerable partition contains exactly 45 queries with verified positives,
    and 5 unanswerable negative controls (e.g. Capital One mortgages).
    """
    partitions = get_query_partitions()
    answerable = partitions["answerable"]
    unanswerable = partitions["unanswerable"]

    assert len(answerable) == 45, f"Expected 45 answerable queries, got {len(answerable)}"
    assert len(unanswerable) == 5, f"Expected 5 unanswerable queries, got {len(unanswerable)}"

    expected_unanswerable = {"q011", "q029", "q030", "q037", "q038"}
    assert set(unanswerable) == expected_unanswerable

    # Check that every answerable query has at least one strictly positive document
    binary_map = get_relevance_map(graded=False)
    for qid in answerable:
        scores = binary_map[qid]
        num_positives = sum(1 for s in scores.values() if s == 1)
        assert num_positives >= 1, f"Answerable query {qid} has 0 positive documents"


def test_audit_golden_set_report() -> None:
    """Verifies that audit_golden_set runs cleanly without error and reports correct statistics."""
    stats = audit_golden_set()
    assert stats["total_queries"] == 50
    assert stats["total_judgements"] >= 1800
    assert stats["missing_queries"] == []
    assert stats["label_counts"]["relevant"] > 0
    assert stats["label_counts"]["partially_relevant"] > 0
    assert stats["label_counts"]["not_relevant"] > 0
