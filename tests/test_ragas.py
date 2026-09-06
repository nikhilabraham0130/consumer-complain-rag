"""
Unit Tests for RAGAS Generation Quality Schemas and Metrics.
"""

import pytest
from pydantic import ValidationError

from evals.ragas_schemas import (
    AnswerRelevanceResponse,
    AtomicClaimVerdict,
    ClaimExtractionResponse,
    ClaimVerificationResponse,
    QueryRagasResult,
    RagasBenchmarkSummary,
)


def test_claim_extraction_cleaning() -> None:
    """Tests that claims are properly cleaned and whitespace stripped."""
    raw_data = {
        "claims": [
            "  Wells Fargo charged an illegal $35 fee.  ",
            "The customer dispute was denied after 10 days.",
            "",
            "   ",
        ]
    }
    resp = ClaimExtractionResponse.model_validate(raw_data)
    assert len(resp.claims) == 2
    assert resp.claims[0] == "Wells Fargo charged an illegal $35 fee."
    assert resp.claims[1] == "The customer dispute was denied after 10 days."


def test_atomic_claim_verdict_valid() -> None:
    """Verifies creation and serialization of an atomic claim verdict."""
    verdict = AtomicClaimVerdict(
        claim="Chase closed the account without prior written notice.",
        is_grounded=True,
        supporting_complaint_ids=["1029384"],
        reasoning="Complaint #1029384 explicitly states account was closed with 0 days notice.",
    )
    assert verdict.is_grounded is True
    assert verdict.supporting_complaint_ids == ["1029384"]


def test_answer_relevance_bounds() -> None:
    """Tests score boundaries: must be between 0.0 and 1.0."""
    valid = AnswerRelevanceResponse(
        relevance_score=0.88,
        is_complete=True,
        reasoning="Directly answers the overdraft inquiry.",
    )
    assert valid.relevance_score == 0.88

    with pytest.raises(ValidationError):
        AnswerRelevanceResponse(
            relevance_score=1.5,
            is_complete=True,
            reasoning="Score exceeds 1.0",
        )

    with pytest.raises(ValidationError):
        AnswerRelevanceResponse(
            relevance_score=-0.2,
            is_complete=False,
            reasoning="Score below 0.0",
        )


def test_query_ragas_result_faithfulness_math() -> None:
    """Tests that faithfulness score accurately matches grounded/total ratio."""
    result = QueryRagasResult(
        query_id="q001",
        query="Wells Fargo overdraft fee disputes",
        executive_summary="Summary text...",
        total_claims=4,
        grounded_claims=3,
        faithfulness=0.75,
        answer_relevance=0.90,
        citation_precision=1.0,
        claims_detail=[
            AtomicClaimVerdict(
                claim="Claim 1",
                is_grounded=True,
                supporting_complaint_ids=["c1"],
                reasoning="Found in c1",
            ),
            AtomicClaimVerdict(
                claim="Claim 2",
                is_grounded=True,
                supporting_complaint_ids=["c2"],
                reasoning="Found in c2",
            ),
            AtomicClaimVerdict(
                claim="Claim 3",
                is_grounded=True,
                supporting_complaint_ids=["c3"],
                reasoning="Found in c3",
            ),
            AtomicClaimVerdict(
                claim="Claim 4",
                is_grounded=False,
                supporting_complaint_ids=[],
                reasoning="Not mentioned in any context doc",
            ),
        ],
    )
    assert result.faithfulness == 0.75
    assert result.grounded_claims / result.total_claims == 0.75
    assert len(result.claims_detail) == 4


def test_ragas_benchmark_summary_aggregation() -> None:
    """Tests benchmark aggregation math and hallucination rate."""
    summary = RagasBenchmarkSummary(
        total_queries=20,
        mean_faithfulness=0.925,
        mean_answer_relevance=0.890,
        mean_citation_precision=1.0,
        hallucination_rate=0.075,
    )
    assert summary.total_queries == 20
    assert summary.mean_faithfulness == 0.925
    assert round(1.0 - summary.mean_faithfulness, 3) == summary.hallucination_rate
