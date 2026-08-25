"""
Unit tests for CFPB API Ingestion Client and Orchestration Pipeline.
Uses pytest-mock to test pagination, retry policies, and error handling without live network calls.
"""

from pathlib import Path
from typing import Any, Dict
import pytest
import requests

from src.ingestion.cfpb_client import CFPBIngestionClient
from src.schemas import CFPBAPIError


def test_fetch_page_success(mocker: Any, sample_api_response_envelope: Dict[str, Any]) -> None:
    """Tests successful fetching and parsing of a single page of API results."""
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_api_response_envelope
    
    client = CFPBIngestionClient()
    mocker.patch.object(client.session, "get", return_value=mock_response)
    
    records = client.fetch_page(size=2, from_offset=0)
    
    assert len(records) == 2
    assert records[0].complaint_id == "5829104"
    assert records[1].complaint_id == "5829105"


def test_fetch_page_http_error_raises_cfpb_api_error(mocker: Any) -> None:
    """Tests that HTTP 500 or network error raises custom CFPBAPIError."""
    client = CFPBIngestionClient()
    mocker.patch.object(
        client.session,
        "get",
        side_effect=requests.exceptions.HTTPError("500 Server Error")
    )
    
    with pytest.raises(CFPBAPIError, match="Failed to fetch data"):
        client.fetch_page(size=10, from_offset=0)


def test_ingest_and_process_pipeline(
    mocker: Any,
    sample_api_response_envelope: Dict[str, Any],
    tmp_path: Path
) -> None:
    """Tests full ingestion loop from mock API hits to Parquet file saving."""
    client = CFPBIngestionClient()
    
    # Mock fetch_page to return sources
    raw_sources = [
        mocker.MagicMock(
            complaint_id="5829104",
            date_received="2023-11-14",
            product="Checking or savings account",
            sub_product="Checking account",
            issue="Unauthorized transaction",
            sub_issue="Wire fraud",
            company="JPMORGAN CHASE & CO.",
            state="IL",
            zip_code="606XX",
            submitted_via="Web",
            company_response_to_consumer="Closed with monetary relief",
            timely="Yes",
            consumer_disputed="No",
            consumer_complaint_narrative="On XX/XX/2023 I discovered an unauthorized charge of {$450.00} on my checking account and Chase refused to refund it."
        )
    ]
    mocker.patch.object(client, "fetch_page", return_value=raw_sources)
    
    output_parquet = tmp_path / "test_complaints.parquet"
    
    df = client.ingest_and_process(
        max_records=1,
        batch_size=1,
        output_file=output_parquet
    )
    
    # Verify DataFrame and Parquet output
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]["complaint_id"] == "5829104"
    assert bool(df.iloc[0]["has_monetary_relief"]) is True
    assert "[DATE]" in df.iloc[0]["cleaned_narrative"]
    assert output_parquet.exists()
