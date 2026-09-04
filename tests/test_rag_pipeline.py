"""
Unit Tests for Phase 5 RAG Pipeline: Citation Guardrails, Wilson Score Math, and Schemas.
"""

import pytest

from src.rag_pipeline import (
    CitationGuardrail,
    ComplianceReport,
    ExtractedFilter,
    PipelineTelemetry,
    ResolutionPrediction,
    ResolutionPredictor,
)


# =============================================================================
# Citation Guardrail Tests
# =============================================================================

def test_citation_guardrail_perfect_precision() -> None:
    """Tests that valid citations matching retrieved context return 100% precision."""
    text = (
        "Chase bank engaged in deceptive fee practices [Complaint #7888421], "
        "which led to unauthorized charges being assessed [Complaint #7855491]."
    )
    retrieved_ids = {"7888421", "7855491", "1234567"}

    verified, unverified, precision = CitationGuardrail.verify_citations(text, retrieved_ids)

    assert set(verified) == {"7888421", "7855491"}
    assert unverified == []
    assert precision == 1.0


def test_citation_guardrail_catches_hallucination() -> None:
    """Tests that hallucinated complaint IDs are correctly flagged and precision drops."""
    text = (
        "The customer disputed an escrow shortage [Complaint #7888421]. "
        "Another customer filed a lawsuit [Complaint #9999999]."
    )
    retrieved_ids = {"7888421"}  # 9999999 was never retrieved!

    verified, unverified, precision = CitationGuardrail.verify_citations(text, retrieved_ids)

    assert verified == ["7888421"]
    assert unverified == ["9999999"]
    assert precision == 0.5


def test_citation_guardrail_no_citations() -> None:
    """Tests handling of text without citation tags."""
    text = "General market observations with no specific complaint citations."
    retrieved_ids = {"7888421"}

    verified, unverified, precision = CitationGuardrail.verify_citations(text, retrieved_ids)

    assert verified == []
    assert unverified == []
    assert precision == 1.0


# =============================================================================
# Resolution Predictor (Wilson Score CI) Tests
# =============================================================================

def test_resolution_predictor_empty_complaints() -> None:
    """Tests edge case when 0 complaints are provided."""
    prediction = ResolutionPredictor.predict([])

    assert prediction.relief_rate == 0.0
    assert prediction.sample_size == 0
    assert prediction.confidence_interval_95 == (0.0, 0.0)


def test_resolution_predictor_wilson_score_math() -> None:
    """
    Tests statistical validity of Wilson Score confidence interval.
    For n=10, k=2: p_hat = 0.20, Wilson CI roughly [0.057, 0.510].
    """
    complaints = [
        {"has_monetary_relief": True},
        {"has_monetary_relief": True},
    ] + [{"has_monetary_relief": False}] * 8

    prediction = ResolutionPredictor.predict(complaints)

    assert prediction.sample_size == 10
    assert prediction.relief_count == 2
    assert prediction.relief_rate == 0.2

    low, high = prediction.confidence_interval_95
    assert 0.0 <= low < 0.2
    assert 0.2 < high <= 1.0
    assert "20.0% monetary relief rate based on n=10" in prediction.interpretation


def test_resolution_predictor_zero_relief() -> None:
    """Tests Wilson score boundary behavior when relief count is 0."""
    complaints = [{"has_monetary_relief": False}] * 20

    prediction = ResolutionPredictor.predict(complaints)

    assert prediction.relief_rate == 0.0
    low, high = prediction.confidence_interval_95
    assert low == 0.0
    assert high > 0.0  # Wilson upper bound is strictly positive even when k=0


# =============================================================================
# Schema Validation Tests
# =============================================================================

def test_compliance_report_schema() -> None:
    """Verifies that ComplianceReport model instantiates with full telemetry and metadata."""
    report = ComplianceReport(
        query="Test query",
        extracted_filter=ExtractedFilter(
            target_company="JPMORGAN CHASE & CO.",
            product_family="Credit card",
            semantic_query="unexpected annual fee",
        ),
        executive_summary="Summary text [Complaint #12345].",
        key_findings=["Finding 1 [Complaint #12345]"],
        risk_level="HIGH",
        cited_complaint_ids=["12345"],
        unverified_citations=[],
        citation_precision=1.0,
        resolution_prediction=ResolutionPrediction(
            relief_rate=0.25,
            confidence_interval_95=(0.10, 0.45),
            sample_size=20,
            relief_count=5,
            interpretation="25% relief rate",
        ),
        telemetry=PipelineTelemetry(
            extraction_time_ms=120.5,
            retrieval_time_ms=45.2,
            generation_time_ms=850.1,
            total_time_ms=1015.8,
        ),
    )

    assert report.risk_level == "HIGH"
    assert report.citation_precision == 1.0
    assert report.telemetry.total_time_ms > 1000

