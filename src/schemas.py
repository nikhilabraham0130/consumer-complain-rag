"""
Pydantic Data Models and Schemas for the CFPB Consumer Complaint Intelligence System.
Defines strict data contracts, field validation rules, and custom exception types.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class CFPBBaseError(Exception):
    """Base exception for all CFPB pipeline operations."""
    pass


class CFPBDataError(CFPBBaseError):
    """Raised when raw data fails integrity or structure checks."""
    pass


class SchemaValidationError(CFPBBaseError):
    """Raised when record fails Pydantic schema validation."""
    pass


class CFPBAPIError(CFPBBaseError):
    """Raised when communication with the CFPB Public API fails."""
    pass


# =============================================================================
# Raw API Record Models (Maps to CFPB Elasticsearch JSON Response)
# =============================================================================

class CFPBRawSource(BaseModel):
    """Represents the raw '_source' payload from the CFPB Open Data API."""
    
    complaint_id: str
    date_received: str
    product: str
    sub_product: Optional[str] = None
    issue: str
    sub_issue: Optional[str] = None
    consumer_complaint_narrative: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("consumer_complaint_narrative", "complaint_what_happened"),
    )
    company: str
    state: Optional[str] = None
    zip_code: Optional[str] = None
    submitted_via: Optional[str] = None
    company_response_to_consumer: Optional[str] = Field(
        default="In progress",
        validation_alias=AliasChoices("company_response_to_consumer", "company_response"),
    )
    timely: Optional[str] = "Yes"
    consumer_disputed: Optional[str] = Field(
        default="N/A",
        validation_alias=AliasChoices("consumer_disputed", "consumer_disputed?"),
    )

    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }


class CFPBApiHit(BaseModel):
    """Represents an individual hit wrapper in Elasticsearch API response."""
    index: Optional[str] = Field(default=None, alias="_index")
    id: Optional[str] = Field(default=None, alias="_id")
    source: CFPBRawSource = Field(..., alias="_source")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore"
    }


class CFPBApiResponse(BaseModel):
    """Top-level response envelope from CFPB complaints API endpoint."""
    hits: Dict[str, Any]

    def extract_sources(self) -> List[CFPBRawSource]:
        """Extracts list of CFPBRawSource objects from the nested 'hits.hits' structure."""
        hit_list = self.hits.get("hits", [])
        extracted = []
        for hit in hit_list:
            source_data = hit.get("_source", hit)
            extracted.append(CFPBRawSource.model_validate(source_data))
        return extracted


# =============================================================================
# Processed Complaint Model (Cleaned, Typed, Ready for Parquet & Embeddings)
# =============================================================================

class ProcessedComplaint(BaseModel):
    """
    Validated, typed, and normalized complaint record.
    Used for Parquet storage, metadata filtering, and RAG vector indexing.
    """
    
    complaint_id: str = Field(..., description="Unique CFPB complaint ID (e.g. '5829104')")
    date_received: date = Field(..., description="Date complaint was received by CFPB")
    product: str = Field(..., min_length=1, description="Primary financial product category")
    sub_product: Optional[str] = Field(default="Not Specified", description="Specific sub-product")
    issue: str = Field(..., min_length=1, description="Primary issue category")
    sub_issue: Optional[str] = Field(default="Not Specified", description="Specific sub-issue")
    company: str = Field(..., min_length=1, description="Standardized uppercase company name")
    state: Optional[str] = Field(default="US", description="2-letter US state abbreviation")
    zip_code: Optional[str] = Field(default=None, description="Masked or 5-digit zip code")
    submitted_via: str = Field(default="Web", description="Submission channel (Web, Referral, Phone)")
    company_response: str = Field(..., description="Official resolution provided by the company")
    timely_response: bool = Field(default=True, description="Whether response was submitted within legal timeframe")
    consumer_disputed: Optional[bool] = Field(default=None, description="Whether consumer formally disputed response")
    raw_narrative: str = Field(..., min_length=20, description="Original uncleaned narrative preserved for audit")
    cleaned_narrative: str = Field(..., min_length=20, description="Scrubbed, normalized narrative for semantic embeddings")
    has_monetary_relief: bool = Field(default=False, description="Derived feature: true if resolution included monetary compensation")
    word_count: int = Field(default=0, ge=0, description="Word count of cleaned narrative")
    char_count: int = Field(default=0, ge=0, description="Character count of cleaned narrative")

    # -------------------------------------------------------------------------
    # Field Validators
    # -------------------------------------------------------------------------

    @field_validator("company", mode="before")
    @classmethod
    def normalize_company(cls, v: Any) -> str:
        """Strip whitespace and convert company name to uppercase."""
        if not v or not isinstance(v, str):
            return "UNKNOWN COMPANY"
        return v.strip().upper()

    @field_validator("date_received", mode="before")
    @classmethod
    def parse_date(cls, v: Any) -> date:
        """Parse various date formats (ISO timestamp, YYYY-MM-DD) into date object."""
        if isinstance(v, date):
            return v
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, str):
            # Handles '2023-11-14T00:00:00-05:00' or '2023-11-14'
            clean_str = v.split("T")[0].strip()
            return datetime.strptime(clean_str, "%Y-%m-%d").date()
        raise ValueError(f"Invalid date format: {v}")

    @field_validator("timely_response", mode="before")
    @classmethod
    def parse_timely(cls, v: Any) -> bool:
        """Converts 'Yes'/'No' strings or booleans to strict bool."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("yes", "true", "1")
        return True

    @field_validator("consumer_disputed", mode="before")
    @classmethod
    def parse_disputed(cls, v: Any) -> Optional[bool]:
        """Converts 'Yes'/'No'/'N/A'/None to Optional[bool]."""
        if v is None or v == "" or v == "N/A":
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            val = v.strip().lower()
            if val in ("yes", "true"):
                return True
            if val in ("no", "false"):
                return False
        return None

    # -------------------------------------------------------------------------
    # Model Validator (Derived Features)
    # -------------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def compute_derived_fields(cls, values: Any) -> Any:
        """Computes derived flags like has_monetary_relief and length stats."""
        if isinstance(values, dict):
            # Accurately isolate monetary relief while excluding non-monetary relief
            response = str(values.get("company_response", "")).lower().strip()
            values["has_monetary_relief"] = ("monetary relief" in response) and ("non-monetary" not in response)
            
            # Compute word and char counts
            cleaned = str(values.get("cleaned_narrative", ""))
            values["char_count"] = len(cleaned)
            values["word_count"] = len(cleaned.split())
            
        return values

    @classmethod
    def from_raw_source(
        cls,
        source: CFPBRawSource,
        cleaned_text: str
    ) -> "ProcessedComplaint":
        """
        Factory method to convert a raw API source object and cleaned text into a ProcessedComplaint.
        """
        if not source.consumer_complaint_narrative:
            raise SchemaValidationError(f"Complaint {source.complaint_id} missing raw narrative.")
            
        return cls(
            complaint_id=str(source.complaint_id),
            date_received=source.date_received,
            product=source.product,
            sub_product=source.sub_product or "Not Specified",
            issue=source.issue,
            sub_issue=source.sub_issue or "Not Specified",
            company=source.company,
            state=source.state or "US",
            zip_code=source.zip_code,
            submitted_via=source.submitted_via or "Web",
            company_response=source.company_response_to_consumer or "Closed with explanation",
            timely_response=source.timely,
            consumer_disputed=source.consumer_disputed,
            raw_narrative=source.consumer_complaint_narrative,
            cleaned_narrative=cleaned_text,
        )
