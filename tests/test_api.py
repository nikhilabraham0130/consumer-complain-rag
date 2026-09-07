"""
Unit and integration tests for FastAPI REST API endpoints.

Tests:
1. Root endpoint metadata & links
2. Health check endpoint (liveness & readiness)
3. Benchmark metrics endpoint (Phase 6 retrieval + Phase 7 RAGAS scores)
4. Golden benchmark queries catalog (50 queries)
5. Query inference endpoint (with mocked RAG pipeline & dependency injection)
6. Query input validation (min/max length, bounds checking)
7. Service unavailable handling when pipeline is uninitialized
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_pipeline
from src.rag_pipeline import (
    ComplianceReport,
    ExtractedFilter,
    PipelineTelemetry,
    ResolutionPrediction,
)


@pytest.fixture
def client():
    """Returns a TestClient with no pre-warmed background pipeline."""
    app.state.pipeline = None
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_compliance_report():
    """Generates a canonical mock ComplianceReport for testing."""
    return ComplianceReport(
        query="Wells Fargo checking account fees",
        extracted_filter=ExtractedFilter(
            target_company="WELLS FARGO & COMPANY",
            product_family="Checking or savings account",
            semantic_query="unauthorized checking account maintenance fees",
            date_min=None,
            date_max=None,
        ),
        executive_summary="Multiple consumers reported unexpected monthly maintenance fees assessed without disclosure [Complaint #1234567].",
        key_findings=[
            "Consumers dispute recurring $12 monthly maintenance fees.",
            "Fee waivers were not honored despite qualifying direct deposits.",
        ],
        risk_level="MEDIUM",
        cited_complaint_ids=["1234567"],
        unverified_citations=[],
        citation_precision=1.0,
        resolution_prediction=ResolutionPrediction(
            relief_rate=0.25,
            confidence_interval_95=(0.12, 0.43),
            sample_size=20,
            relief_count=5,
            interpretation="25.0% historical relief probability (95% CI: [12.0%, 43.0%])",
        ),
        telemetry=PipelineTelemetry(
            extraction_time_ms=120.5,
            retrieval_time_ms=85.2,
            generation_time_ms=1450.0,
            total_time_ms=1655.7,
        ),
    )


def test_root_endpoint(client):
    """Test root endpoint returns status 200 and expected metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"
    assert data["metrics_url"] == "/metrics"


def test_health_endpoint(client):
    """Test health check returns system liveness and index status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "database" in data
    assert "indexes" in data
    assert "pipeline_ready" in data


def test_metrics_endpoint(client):
    """Test metrics endpoint returns Phase 6 and Phase 7 benchmark summaries."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "retrieval_ablation" in data
    assert "ragas_generation" in data
    assert isinstance(data["retrieval_ablation"], list)
    assert data["ragas_generation"]["mean_faithfulness"] >= 0.90
    assert data["ragas_generation"]["mean_citation_precision"] == 1.0


def test_benchmark_queries_endpoint(client):
    """Test benchmark queries endpoint returns catalog of golden queries."""
    response = client.get("/benchmark/queries")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 50
    first = data[0]
    assert "query_id" in first
    assert "query" in first
    assert "company" in first
    assert "product_family" in first


def test_query_endpoint_uninitialized(client):
    """Test query endpoint returns 503 when pipeline is not warmed up."""
    payload = {"query": "Wells Fargo overdraft fee disputes", "top_k": 5}
    response = client.post("/query", json=payload)
    assert response.status_code == 503
    assert "uninitialized" in response.json()["detail"].lower()


def test_query_endpoint_success(client, mock_compliance_report, mocker):
    """Test query endpoint returns full ComplianceReport with mock pipeline."""
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.run.return_value = mock_compliance_report

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline

    payload = {"query": "Wells Fargo checking account fees", "top_k": 5}
    response = client.post("/query", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Wells Fargo checking account fees"
    assert data["risk_level"] == "MEDIUM"
    assert data["citation_precision"] == 1.0
    assert "1234567" in data["cited_complaint_ids"]
    assert data["telemetry"]["total_time_ms"] == 1655.7
    mock_pipeline.run.assert_called_once_with("Wells Fargo checking account fees")


def test_query_endpoint_validation_error(client, mocker):
    """Test query endpoint returns 422 for invalid payloads."""
    app.dependency_overrides[get_pipeline] = lambda: mocker.MagicMock()

    # Query too short (< 3 chars)
    response = client.post("/query", json={"query": "hi"})
    assert response.status_code == 422

    # Top-k out of bounds (> 20)
    response = client.post("/query", json={"query": "valid query here", "top_k": 50})
    assert response.status_code == 422
