"""
Unit tests for CFPB API Ingestion Client and Orchestration Pipeline.
Uses pytest-mock to test cursor pagination, quota balancing, deduplication,
retry policies, and error handling without live network calls.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pytest
import requests

from src.ingestion.cfpb_client import CFPBIngestionClient
from src.schemas import CFPBAPIError, CFPBRawSource

NARRATIVE = (
    "On XX/XX/2023 I discovered an unauthorized charge of {$450.00} on my checking "
    "account and the bank refused to refund it despite repeated written requests."
)


def make_source(
    complaint_id: str = "5829104",
    company: str = "JPMORGAN CHASE & CO.",
    product: str = "Checking or savings account",
    narrative: str = NARRATIVE,
) -> CFPBRawSource:
    """Builds a realistic raw API source object for pipeline tests."""
    return CFPBRawSource(
        complaint_id=complaint_id,
        date_received="2023-11-14T00:00:00-05:00",
        product=product,
        sub_product="Checking account",
        issue="Unauthorized transaction",
        sub_issue="Wire fraud",
        company=company,
        state="IL",
        zip_code="606XX",
        submitted_via="Web",
        company_response="Closed with monetary relief",
        timely="Yes",
        consumer_complaint_narrative=narrative,
    )


# =============================================================================
# fetch_page
# =============================================================================

def test_fetch_page_success(mocker: Any, sample_api_response_envelope: Dict[str, Any]) -> None:
    """Tests successful fetching and parsing of a single page of API results."""
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_api_response_envelope

    client = CFPBIngestionClient()
    mocker.patch.object(client.session, "get", return_value=mock_response)

    records, _cursor = client.fetch_page(size=2)

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
        client.fetch_page(size=10)


def test_fetch_page_sends_repeated_product_params(
    mocker: Any, sample_api_response_envelope: Dict[str, Any]
) -> None:
    """
    A product family maps to several CFPB product spellings; each must be sent as
    its own `product` param so the API OR-s them together.
    """
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = sample_api_response_envelope
    client = CFPBIngestionClient()
    spy = mocker.patch.object(client.session, "get", return_value=mock_response)

    client.fetch_page(
        size=10,
        company="CITIBANK, N.A.",
        products=["Credit card", "Credit card or prepaid card"],
    )

    sent = spy.call_args.kwargs["params"]
    assert ("company", "CITIBANK, N.A.") in sent
    assert ("product", "Credit card") in sent
    assert ("product", "Credit card or prepaid card") in sent
    assert sum(1 for k, _ in sent if k == "product") == 2


def test_fetch_page_uses_cursor_not_offset(
    mocker: Any, sample_api_response_envelope: Dict[str, Any]
) -> None:
    """
    Regression guard: the CFPB API validates `frm` but ignores it, so offset
    paging silently returns page 1 forever. Only `search_after` may be sent.
    """
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = sample_api_response_envelope
    client = CFPBIngestionClient()
    spy = mocker.patch.object(client.session, "get", return_value=mock_response)

    client.fetch_page(size=10, search_after="1.0_8094492")

    keys = [k for k, _ in spy.call_args.kwargs["params"]]
    assert "frm" not in keys and "from" not in keys
    assert ("search_after", "1.0_8094492") in spy.call_args.kwargs["params"]


def test_fetch_page_extracts_next_cursor(mocker: Any) -> None:
    """The cursor for the next page is '<score>_<complaint_id>' of the last hit."""
    envelope = {
        "hits": {
            "total": {"value": 2, "relation": "eq"},
            "hits": [
                {"_score": 1.0, "_source": {
                    "complaint_id": "111", "date_received": "2023-01-01",
                    "product": "Mortgage", "issue": "x", "company": "C"}},
                {"_score": 1.0, "_source": {
                    "complaint_id": "222", "date_received": "2023-01-02",
                    "product": "Mortgage", "issue": "x", "company": "C"}},
            ],
        }
    }
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = envelope
    client = CFPBIngestionClient()
    mocker.patch.object(client.session, "get", return_value=mock_response)

    _records, cursor = client.fetch_page(size=2)
    assert cursor == "1.0_222"


# =============================================================================
# Record validation (defence in depth against fuzzy server-side matches)
# =============================================================================

def test_build_record_rejects_mismatched_company() -> None:
    """A record from the wrong company must never contaminate a cell."""
    source = make_source(company="WELLS FARGO & COMPANY")
    assert CFPBIngestionClient._build_record(
        source, "JPMORGAN CHASE & CO.", ["Checking or savings account"]
    ) is None


def test_build_record_rejects_mismatched_product() -> None:
    """A record outside the cell's product family must be skipped."""
    source = make_source(product="Mortgage")
    assert CFPBIngestionClient._build_record(
        source, "JPMORGAN CHASE & CO.", ["Checking or savings account"]
    ) is None


def test_build_record_rejects_short_narrative() -> None:
    """Narratives below the minimum word count are filtered out."""
    source = make_source(narrative="Bank charged me a fee and I want it back now.")
    assert CFPBIngestionClient._build_record(
        source, "JPMORGAN CHASE & CO.", ["Checking or savings account"]
    ) is None


def test_build_record_accepts_valid_source() -> None:
    """A matching, sufficiently long record is converted and PII-normalized."""
    record = CFPBIngestionClient._build_record(
        make_source(), "JPMORGAN CHASE & CO.", ["Checking or savings account"]
    )
    assert record is not None
    assert record.complaint_id == "5829104"
    assert record.has_monetary_relief is True
    assert "[DATE]" in record.cleaned_narrative


# =============================================================================
# Balanced ingestion pipeline
# =============================================================================

def paging_fetch(pages: int = 100):
    """Builds a fake fetch_page that yields fresh records and an advancing cursor."""
    def _fetch(size, search_after, company, products, has_narrative=True):
        page = 0 if search_after is None else int(search_after.split("_")[-1]) + 1
        if page >= pages:
            return [], None
        sources = [
            make_source(
                complaint_id=f"{company[:4]}-{products[0][:4]}-{page}-{i}",
                company=company,
                product=products[0],
            )
            for i in range(size)
        ]
        return sources, f"1.0_{page}"
    return _fetch


def test_ingest_and_process_balances_cells(mocker: Any, tmp_path: Path) -> None:
    """
    Each company x product family cell must receive an equal quota, so a
    high-volume institution cannot dominate the corpus.
    """
    mocker.patch("src.ingestion.cfpb_client.time.sleep")
    client = CFPBIngestionClient()
    mocker.patch.object(client, "fetch_page", side_effect=paging_fetch())

    df = client.ingest_and_process(
        max_records=40,
        batch_size=10,
        companies=["JPMORGAN CHASE & CO.", "CITIBANK, N.A."],
        product_families={
            "Checking or savings account": ["Checking or savings account"],
            "Mortgage": ["Mortgage"],
        },
        output_file=tmp_path / "balanced.parquet",
    )

    assert len(df) == 40
    counts = df.groupby(["company", "product"]).size()
    assert len(counts) == 4, "expected 2 companies x 2 families = 4 cells"
    assert set(counts.unique()) == {10}, f"cells should be equal, got {counts.to_dict()}"


def test_ingest_emits_no_duplicate_complaint_ids(mocker: Any, tmp_path: Path) -> None:
    """
    Regression test: offset paging used to repeat page 1, so the same complaint
    was appended many times. Every emitted complaint_id must be unique.
    """
    mocker.patch("src.ingestion.cfpb_client.time.sleep")
    client = CFPBIngestionClient()
    mocker.patch.object(client, "fetch_page", side_effect=paging_fetch())

    df = client.ingest_and_process(
        max_records=60,
        batch_size=10,
        companies=["JPMORGAN CHASE & CO."],
        product_families={"Mortgage": ["Mortgage"]},
        output_file=tmp_path / "dedup.parquet",
    )

    assert not df.empty
    assert df["complaint_id"].duplicated().sum() == 0


def test_repeated_page_is_deduplicated(mocker: Any, tmp_path: Path) -> None:
    """
    Even if the API misbehaves and serves the same records under a new cursor,
    the global seen-set must prevent duplicates from reaching the corpus.
    """
    mocker.patch("src.ingestion.cfpb_client.time.sleep")
    client = CFPBIngestionClient()

    counter = {"n": 0}

    def always_same(size, search_after, company, products, has_narrative=True):
        counter["n"] += 1
        # Same 3 complaint IDs every call, but an ever-changing cursor.
        return (
            [make_source(complaint_id=str(i), company=company, product=products[0])
             for i in range(3)],
            f"1.0_{counter['n']}",
        )

    mocker.patch.object(client, "fetch_page", side_effect=always_same)

    df = client.ingest_and_process(
        max_records=50,
        batch_size=3,
        companies=["JPMORGAN CHASE & CO."],
        product_families={"Mortgage": ["Mortgage"]},
        output_file=tmp_path / "repeat.parquet",
    )

    assert len(df) == 3, "only the 3 distinct complaints should survive"
    assert df["complaint_id"].duplicated().sum() == 0


def test_ingest_and_process_writes_parquet(mocker: Any, tmp_path: Path) -> None:
    """Tests full ingestion loop from mock API hits to Parquet file saving."""
    mocker.patch("src.ingestion.cfpb_client.time.sleep")
    client = CFPBIngestionClient()
    mocker.patch.object(client, "fetch_page", return_value=([make_source()], None))

    out = tmp_path / "test_complaints.parquet"
    df = client.ingest_and_process(
        max_records=1,
        batch_size=1,
        companies=["JPMORGAN CHASE & CO."],
        product_families={"Checking or savings account": ["Checking or savings account"]},
        output_file=out,
    )

    assert len(df) == 1
    assert df.iloc[0]["complaint_id"] == "5829104"
    assert bool(df.iloc[0]["has_monetary_relief"]) is True
    assert "[DATE]" in df.iloc[0]["cleaned_narrative"]
    assert out.exists()


def test_pagination_guard_stops_unfillable_cell(mocker: Any, tmp_path: Path) -> None:
    """
    A quota that can never be filled must terminate at MAX_PAGES_PER_CELL rather
    than paging through the entire 1.69M-row corpus.
    """
    mocker.patch("src.ingestion.cfpb_client.time.sleep")
    mocker.patch("src.ingestion.cfpb_client.settings.MAX_PAGES_PER_CELL", 5)
    client = CFPBIngestionClient()

    counter = {"n": 0}

    def wrong_company(size, search_after, company, products, has_narrative=True):
        counter["n"] += 1
        return (
            [make_source(complaint_id=f"x{counter['n']}", company="WELLS FARGO & COMPANY")],
            f"1.0_{counter['n']}",
        )

    fetch = mocker.patch.object(client, "fetch_page", side_effect=wrong_company)

    df = client.ingest_and_process(
        max_records=100,
        batch_size=10,
        companies=["JPMORGAN CHASE & CO."],
        product_families={"Checking or savings account": ["Checking or savings account"]},
        output_file=tmp_path / "empty.parquet",
    )

    assert df.empty
    assert fetch.call_count == 5, "should stop at MAX_PAGES_PER_CELL"


def test_stalled_cursor_breaks_loop(mocker: Any, tmp_path: Path) -> None:
    """A cursor that stops advancing means the API is repeating a page: stop."""
    mocker.patch("src.ingestion.cfpb_client.time.sleep")
    client = CFPBIngestionClient()
    fetch = mocker.patch.object(
        client, "fetch_page",
        return_value=([make_source(company="WELLS FARGO & COMPANY")], None),
    )

    df = client.ingest_and_process(
        max_records=100,
        batch_size=10,
        companies=["JPMORGAN CHASE & CO."],
        product_families={"Checking or savings account": ["Checking or savings account"]},
        output_file=tmp_path / "stall.parquet",
    )

    assert df.empty
    assert fetch.call_count == 1, "a None cursor must end the cell immediately"


def test_empty_api_response_breaks_cell_loop(mocker: Any, tmp_path: Path) -> None:
    """An exhausted cell stops immediately instead of paging to the cap."""
    mocker.patch("src.ingestion.cfpb_client.time.sleep")
    client = CFPBIngestionClient()
    fetch = mocker.patch.object(client, "fetch_page", return_value=([], None))

    df = client.ingest_and_process(
        max_records=100,
        batch_size=10,
        companies=["JPMORGAN CHASE & CO."],
        product_families={"Checking or savings account": ["Checking or savings account"]},
        output_file=tmp_path / "empty.parquet",
    )

    assert df.empty
    assert fetch.call_count == 1
