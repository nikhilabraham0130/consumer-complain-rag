"""
Unit tests for Pydantic schema validation, normalization, and derived field logic.
"""

from datetime import date
from typing import Any, Dict
import pytest

from src.schemas import (
    CFPBApiResponse,
    CFPBRawSource,
    ProcessedComplaint,
    SchemaValidationError,
)


def test_raw_source_parsing(sample_raw_cfpb_hit: Dict[str, Any]) -> None:
    """Tests parsing raw CFPB JSON source with field aliases."""
    raw_source = CFPBRawSource.model_validate(sample_raw_cfpb_hit["_source"])
    
    assert raw_source.complaint_id == "5829104"
    assert raw_source.company == "JPMORGAN CHASE & CO."
    assert raw_source.company_response_to_consumer == "Closed with monetary relief"
    assert raw_source.consumer_disputed == "No"
    assert raw_source.consumer_complaint_narrative is not None
    assert "Regulation E" in raw_source.consumer_complaint_narrative


def test_api_response_envelope_extraction(sample_api_response_envelope: Dict[str, Any]) -> None:
    """Tests extracting individual CFPBRawSource records from the API response envelope."""
    envelope = CFPBApiResponse.model_validate(sample_api_response_envelope)
    sources = envelope.extract_sources()
    
    assert len(sources) == 2
    assert sources[0].complaint_id == "5829104"
    assert sources[1].complaint_id == "5829105"
    assert sources[0].consumer_complaint_narrative is not None
    assert sources[1].consumer_complaint_narrative is None


def test_processed_complaint_normalization(sample_raw_cfpb_hit: Dict[str, Any]) -> None:
    """Tests factory creation of ProcessedComplaint with date parsing and derived fields."""
    raw_source = CFPBRawSource.model_validate(sample_raw_cfpb_hit["_source"])
    cleaned_text = "On [DATE], I discovered an unauthorized charge of [AMOUNT] from [ENTITY] on my checking account."
    
    processed = ProcessedComplaint.from_raw_source(raw_source, cleaned_text=cleaned_text)
    
    # Assert date normalization
    assert processed.date_received == date(2023, 11, 14)
    assert isinstance(processed.date_received, date)
    
    # Assert company normalization
    assert processed.company == "JPMORGAN CHASE & CO."
    
    # Assert boolean parsing
    assert processed.timely_response is True
    assert processed.consumer_disputed is False  # 'No' converts to False
    
    # Assert derived features
    assert processed.has_monetary_relief is True
    assert processed.word_count > 5
    assert processed.char_count == len(cleaned_text)


def test_processed_complaint_missing_narrative_raises_error(sample_raw_cfpb_hit_no_narrative: Dict[str, Any]) -> None:
    """Tests that attempting to process a record without a narrative raises SchemaValidationError."""
    raw_source = CFPBRawSource.model_validate(sample_raw_cfpb_hit_no_narrative["_source"])
    
    with pytest.raises(SchemaValidationError, match="missing raw narrative"):
        ProcessedComplaint.from_raw_source(raw_source, cleaned_text="Some text")


def test_date_parsing_variations() -> None:
    """Tests that both ISO timestamps and plain YYYY-MM-DD strings parse to date objects."""
    base_data = {
        "complaint_id": "1001",
        "product": "Checking or savings account",
        "issue": "Managing an account",
        "company": "wells fargo",
        "company_response": "Closed with explanation",
        "raw_narrative": "A valid complaint narrative for testing purposes.",
        "cleaned_narrative": "A valid complaint narrative for testing purposes.",
    }
    
    # Test with ISO string
    rec1 = ProcessedComplaint.model_validate({**base_data, "date_received": "2024-05-20T14:30:00-04:00"})
    assert rec1.date_received == date(2024, 5, 20)
    
    # Test with simple date string
    rec2 = ProcessedComplaint.model_validate({**base_data, "date_received": "2024-05-20"})
    assert rec2.date_received == date(2024, 5, 20)
    
    # Test company uppercase normalization
    assert rec1.company == "WELLS FARGO"


def test_monetary_relief_flag_computation() -> None:
    """Tests that has_monetary_relief correctly identifies monetary vs non-monetary responses."""
    base_data = {
        "complaint_id": "1002",
        "date_received": "2024-01-01",
        "product": "Mortgage",
        "issue": "Applying for a mortgage",
        "company": "BANK OF AMERICA",
        "raw_narrative": "A valid complaint narrative for testing purposes.",
        "cleaned_narrative": "A valid complaint narrative for testing purposes.",
    }
    
    rec_relief = ProcessedComplaint.model_validate({**base_data, "company_response": "Closed with monetary relief"})
    assert rec_relief.has_monetary_relief is True
    
    rec_explanation = ProcessedComplaint.model_validate({**base_data, "company_response": "Closed with explanation"})
    assert rec_explanation.has_monetary_relief is False
    
    rec_non_monetary = ProcessedComplaint.model_validate({**base_data, "company_response": "Closed with non-monetary relief"})
    assert rec_non_monetary.has_monetary_relief is False
