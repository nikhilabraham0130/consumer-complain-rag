"""
CFPB Open Data API Ingestion Client and Preprocessing Orchestrator.
Fetches, cleans, validates, and serializes consumer complaint narratives into Apache Parquet.
"""

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from src.config import settings
from src.ingestion.cleaner import clean_narrative, is_valid_narrative
from src.schemas import (
    CFPBAPIError,
    CFPBApiResponse,
    CFPBRawSource,
    ProcessedComplaint,
    SchemaValidationError,
)
from src.utils.logger import get_logger

logger = get_logger("cfpb_ingestion")


class CFPBIngestionClient:
    """
    Resilient client for querying the official CFPB Consumer Complaint database API.
    Handles pagination, retry backoff, text sanitization, and Parquet caching.
    """

    DEFAULT_HEADERS = {
        "User-Agent": "ConsumerComplaintIntelligenceRAG/1.0 (Purdue University Academic Project; mailto:student@purdue.edu)",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
    }

    def __init__(
        self,
        base_url: str = settings.CFPB_API_BASE_URL,
        timeout: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.session = self._build_resilient_session(max_retries, backoff_factor)

    @classmethod
    def _build_resilient_session(cls, max_retries: int, backoff_factor: float) -> requests.Session:
        """Configures a requests Session with automatic exponential backoff retries and browser headers."""
        session = requests.Session()
        session.headers.update(cls.DEFAULT_HEADERS)
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch_page(
        self,
        size: int = 50,
        from_offset: int = 0,
        company: Optional[str] = None,
        has_narrative: bool = True,
    ) -> List[CFPBRawSource]:
        """
        Fetches a single paginated batch of raw complaint records from the CFPB API.

        Args:
            size: Number of records to fetch per page (optimal: 25-50)
            from_offset: Offset pagination index
            company: Optional specific company name to filter
            has_narrative: Whether to restrict results to complaints with text narratives

        Returns:
            List of validated CFPBRawSource objects.

        Raises:
            CFPBAPIError: If HTTP request or JSON decoding fails.
        """
        params: Dict[str, Any] = {
            "size": size,
            "from": from_offset,
            "has_narrative": "true" if has_narrative else "false",
            "date_received_max": "2023-12-31",  # Skip recent complaints that lack published narratives
        }

        if company:
            params["company"] = company

        try:
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CFPB API request failed at offset={from_offset}, company={company}: {e}")
            raise CFPBAPIError(f"Failed to fetch data from CFPB API: {e}") from e
        except ValueError as e:
            logger.error(f"Failed to decode CFPB API JSON response: {e}")
            raise CFPBAPIError(f"Invalid JSON response from CFPB API: {e}") from e

        # Validate response envelope structure with Pydantic
        envelope = CFPBApiResponse.model_validate(data)
        return envelope.extract_sources()

    def ingest_and_process(
        self,
        max_records: int = 200,
        batch_size: int = 100,
        companies: Optional[List[str]] = None,
        output_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Orchestrates end-to-end ingestion pipeline.
        Fetches directly from the live feed and filters target companies locally 
        to avoid Elasticsearch API parsing errors on special characters (like '&').

        Args:
            max_records: Target total number of processed records to ingest
            batch_size: Number of records per API call (default 100)
            companies: Target company list
            output_file: Output path for Parquet file

        Returns:
            pandas.DataFrame of validated, cleaned complaints.
        """
        target_companies = set(companies or settings.TARGET_COMPANIES)
        dest_file = output_file or settings.PROCESSED_PARQUET_FILE
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[*] Starting Ingestion: Target={max_records} records across {len(target_companies)} institutions (Live Stream Mode)"
        )

        processed_records: List[ProcessedComplaint] = []
        from_offset = 0

        while len(processed_records) < max_records:
            try:
                raw_sources = self.fetch_page(
                    size=batch_size,
                    from_offset=from_offset,
                    company=None,  # Fetch raw stream without URL filtering!
                    has_narrative=True,
                )
            except CFPBAPIError as e:
                logger.error(f"API Error fetching feed: {e}")
                break

            if not raw_sources:
                logger.warning("No more records returned from API.")
                break

            for source in raw_sources:
                # 1. Local string matching is lightning fast and immune to URL encoding bugs
                if source.company.strip().upper() not in target_companies:
                    continue

                if not source.consumer_complaint_narrative:
                    continue

                cleaned_text = clean_narrative(source.consumer_complaint_narrative)
                if not is_valid_narrative(cleaned_text, min_words=15):
                    continue

                try:
                    record = ProcessedComplaint.from_raw_source(
                        source=source,
                        cleaned_text=cleaned_text,
                    )
                    processed_records.append(record)
                except SchemaValidationError:
                    continue

                if len(processed_records) >= max_records:
                    break

            from_offset += batch_size
            time.sleep(0.2)  # Polite API pacing
            
            # Print a progress update every iteration
            logger.info(f"   Collected {len(processed_records)}/{max_records} matching records...")

        if not processed_records:
            logger.warning("No valid complaint records were collected.")
            return pd.DataFrame()

        # Convert list of Pydantic models to Pandas DataFrame
        records_dict = [rec.model_dump() for rec in processed_records]
        df = pd.DataFrame(records_dict)

        # Save to Parquet with PyArrow engine
        df.to_parquet(dest_file, engine="pyarrow", index=False)
        logger.info(
            f"[SUCCESS] Successfully saved {len(df)} complaints to [bold green]{dest_file}[/bold green]"
        )

        return df


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main() -> None:
    """CLI runner to execute data ingestion directly from the terminal."""
    parser = argparse.ArgumentParser(
        description="Ingest and preprocess CFPB complaint narratives for RAG."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of valid complaints to ingest (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(settings.PROCESSED_PARQUET_FILE),
        help="Path to output Parquet file",
    )
    args = parser.parse_args()

    client = CFPBIngestionClient()
    df = client.ingest_and_process(
        max_records=args.limit,
        output_file=Path(args.output),
    )

    if not df.empty:
        logger.info("\n[SUMMARY] Dataset Ingestion Summary:")
        logger.info(f"Total Rows: {len(df)}")
        logger.info(f"Unique Companies: {df['company'].nunique()}")
        logger.info(f"Monetary Relief Rate: {(df['has_monetary_relief'].mean() * 100):.1f}%")
        logger.info(f"Average Narrative Words: {df['word_count'].mean():.0f}")


if __name__ == "__main__":
    main()
