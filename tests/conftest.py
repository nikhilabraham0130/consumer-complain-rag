"""
Pytest Fixtures and Mock Data for Consumer Complaint Intelligence RAG tests.
Provides realistic CFPB API responses, redacted narratives, and edge-case samples.
"""

from typing import Any, Dict
import pytest


@pytest.fixture
def sample_raw_cfpb_hit() -> Dict[str, Any]:
    """A realistic single Elasticsearch hit from the CFPB Public API."""
    return {
        "_index": "complaints-2023",
        "_id": "5829104",
        "_source": {
            "complaint_id": "5829104",
            "date_received": "2023-11-14T00:00:00-05:00",
            "product": "Checking or savings account",
            "sub_product": "Checking account",
            "issue": "Problem with a lender or other company charged your account",
            "sub_issue": "Transaction was not authorized",
            "consumer_complaint_narrative": "On XX/XX/2023, I discovered an unauthorized charge of {$450.00} from XXXX on my checking account. I contacted JPMorgan Chase immediately, but they refused to credit the funds under Regulation E.",
            "company": "JPMORGAN CHASE & CO.",
            "state": "IL",
            "zip_code": "606XX",
            "submitted_via": "Web",
            "company_response_to_consumer": "Closed with monetary relief",
            "timely": "Yes",
            "consumer_disputed?": "No"
        }
    }


@pytest.fixture
def sample_raw_cfpb_hit_no_narrative() -> Dict[str, Any]:
    """A CFPB hit where consumer did not provide an opt-in narrative."""
    return {
        "_index": "complaints-2023",
        "_id": "5829105",
        "_source": {
            "complaint_id": "5829105",
            "date_received": "2023-11-15",
            "product": "Credit card or prepaid card",
            "sub_product": "General-purpose credit card",
            "issue": "Fees or interest",
            "sub_issue": "Problem with fees",
            "consumer_complaint_narrative": None,
            "company": "CITIBANK, N.A.",
            "state": "NY",
            "zip_code": "100XX",
            "submitted_via": "Web",
            "company_response_to_consumer": "Closed with explanation",
            "timely": "Yes",
            "consumer_disputed?": "N/A"
        }
    }


@pytest.fixture
def sample_api_response_envelope(sample_raw_cfpb_hit: Dict[str, Any], sample_raw_cfpb_hit_no_narrative: Dict[str, Any]) -> Dict[str, Any]:
    """A multi-hit CFPB API response envelope."""
    return {
        "hits": {
            "total": {
                "value": 2,
                "relation": "eq"
            },
            "hits": [
                sample_raw_cfpb_hit,
                sample_raw_cfpb_hit_no_narrative
            ]
        }
    }
