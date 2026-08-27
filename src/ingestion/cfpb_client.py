"""
CFPB Open Data API Ingestion Client and Preprocessing Orchestrator.
Fetches, cleans, validates, and serializes consumer complaint narratives into Apache Parquet.

Sampling strategy: the corpus is built as a balanced grid of
`company x product family` cells rather than a raw stream slice. Each cell is
filled by a dedicated server-side filtered query, so no single high-volume
institution (Wells Fargo alone has ~13k eligible complaints) can dominate the
dataset. See .work/deep_dive_plan.md Project 3, section 2.
"""

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
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
        size: int = 100,
        search_after: Optional[str] = None,
        company: Optional[str] = None,
        products: Optional[List[str]] = None,
        has_narrative: bool = True,
    ) -> Tuple[List[CFPBRawSource], Optional[str]]:
        """
        Fetches a single paginated batch of raw complaint records from the CFPB API.

        Args:
            size: Number of records to fetch per page
            search_after: Cursor from the previous page ("<score>_<complaint_id>").
                The CFPB API is cursor-paginated; its `frm` offset parameter is
                validated but ignored, so offset paging silently repeats page 1.
            company: Optional exact company name to filter server-side
            products: Optional list of CFPB product values to filter server-side.
                Repeated `product` params are OR-ed by the API, which lets one
                logical family (e.g. "Credit card" + "Credit card or prepaid card")
                be requested in a single call.
            has_narrative: Whether to restrict results to complaints with text narratives

        Returns:
            Tuple of (validated CFPBRawSource objects, cursor for the next page).

        Raises:
            CFPBAPIError: If HTTP request or JSON decoding fails.
        """
        # A list of tuples (not a dict) so `product` can repeat.
        params: List[Tuple[str, Any]] = [
            ("size", size),
            ("has_narrative", "true" if has_narrative else "false"),
            ("date_received_min", settings.INGEST_DATE_MIN),
            ("date_received_max", settings.INGEST_DATE_MAX),
        ]

        if search_after:
            params.append(("search_after", search_after))
        if company:
            params.append(("company", company))
        for product in products or []:
            params.append(("product", product))

        try:
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"CFPB API request failed for company={company}, cursor={search_after}: {e}")
            raise CFPBAPIError(f"Failed to fetch data from CFPB API: {e}") from e
        except ValueError as e:
            logger.error(f"Failed to decode CFPB API JSON response: {e}")
            raise CFPBAPIError(f"Invalid JSON response from CFPB API: {e}") from e

        # Validate response envelope structure with Pydantic
        envelope = CFPBApiResponse.model_validate(data)
        return envelope.extract_sources(), envelope.next_cursor()

    def _fill_cell(
        self,
        company: str,
        family: str,
        product_values: List[str],
        quota: int,
        batch_size: int,
        remaining_budget: int,
        seen_ids: Set[str],
    ) -> List[ProcessedComplaint]:
        """
        Fills a single `company x product family` cell up to `quota` records.

        Pages via the API's `search_after` cursor. Bounded by
        settings.MAX_PAGES_PER_CELL, and stops early if the cursor stalls or the
        page yields nothing new, so an unfillable quota can never spin.
        """
        target = min(quota, remaining_budget)
        collected: List[ProcessedComplaint] = []
        cursor: Optional[str] = None
        pages = 0

        while len(collected) < target and pages < settings.MAX_PAGES_PER_CELL:
            try:
                sources, next_cursor = self.fetch_page(
                    size=batch_size,
                    search_after=cursor,
                    company=company,
                    products=product_values,
                    has_narrative=True,
                )
            except CFPBAPIError as e:
                logger.error(f"    API error on cell ({company} / {family}) page={pages}: {e}")
                break

            pages += 1
            if not sources:
                break

            added = 0
            for source in sources:
                if source.complaint_id in seen_ids:
                    continue
                record = self._build_record(source, company, product_values)
                if record is None:
                    continue
                seen_ids.add(source.complaint_id)
                collected.append(record)
                added += 1
                if len(collected) >= target:
                    break

            # A cursor that does not advance means the API is repeating a page.
            if next_cursor is None or next_cursor == cursor:
                break
            cursor = next_cursor

            if added == 0 and len(collected) < target:
                # Page held nothing usable; keep paging but do not sleep needlessly.
                pass

            time.sleep(0.2)  # Polite API pacing

        if len(collected) < target:
            logger.warning(
                f"    Cell underfilled: {company} / {family} -> {len(collected)}/{target} "
                f"(exhausted after {pages} pages)"
            )
        return collected

    @staticmethod
    def _build_record(
        source: CFPBRawSource,
        expected_company: str,
        product_values: List[str],
    ) -> Optional[ProcessedComplaint]:
        """
        Validates, cleans, and converts one raw source into a ProcessedComplaint.
        Returns None when the record should be skipped.

        The company/product checks are defence in depth: the API already filters
        server-side, but a fuzzy match would otherwise contaminate a cell.
        """
        if source.company.strip().upper() != expected_company.strip().upper():
            return None
        if source.product not in product_values:
            return None
        if not source.consumer_complaint_narrative:
            return None

        cleaned_text = clean_narrative(source.consumer_complaint_narrative)
        if not is_valid_narrative(cleaned_text, min_words=15):
            return None

        try:
            return ProcessedComplaint.from_raw_source(source=source, cleaned_text=cleaned_text)
        except SchemaValidationError:
            return None

    def ingest_and_process(
        self,
        max_records: int = 10000,
        batch_size: int = 100,
        companies: Optional[List[str]] = None,
        product_families: Optional[Dict[str, List[str]]] = None,
        output_file: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Orchestrates the end-to-end balanced ingestion pipeline.

        Walks every `company x product family` cell, filling each to an equal
        quota via server-side filtered queries, then serializes the result to
        Parquet.

        Args:
            max_records: Total target number of processed records
            batch_size: Number of records per API call
            companies: Target company list (defaults to settings.TARGET_COMPANIES)
            product_families: Family name -> CFPB product values map
            output_file: Output path for Parquet file

        Returns:
            pandas.DataFrame of validated, cleaned complaints.
        """
        target_companies = companies or settings.TARGET_COMPANIES
        families = product_families or settings.TARGET_PRODUCT_FAMILIES
        dest_file = output_file or settings.PROCESSED_PARQUET_FILE
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        cells = [(company, family) for company in target_companies for family in families]
        quota = max(1, max_records // max(len(cells), 1))

        logger.info(
            f"[*] Starting Balanced Ingestion: target={max_records} across "
            f"{len(target_companies)} institutions x {len(families)} product families "
            f"= {len(cells)} cells @ {quota} records/cell"
        )

        processed_records: List[ProcessedComplaint] = []
        underfilled: List[str] = []
        seen_ids: Set[str] = set()  # Global dedup guard across all cells

        for company, family in cells:
            if len(processed_records) >= max_records:
                break

            cell_records = self._fill_cell(
                company=company,
                family=family,
                product_values=families[family],
                quota=quota,
                batch_size=batch_size,
                remaining_budget=max_records - len(processed_records),
                seen_ids=seen_ids,
            )
            processed_records.extend(cell_records)

            if len(cell_records) < quota:
                underfilled.append(f"{company} / {family} ({len(cell_records)}/{quota})")

            logger.info(
                f"   [{company[:28]:28} | {family[:26]:26}] "
                f"+{len(cell_records):>4}  total={len(processed_records)}/{max_records}"
            )

        if not processed_records:
            logger.warning("No valid complaint records were collected.")
            return pd.DataFrame()

        # Convert list of Pydantic models to Pandas DataFrame
        records_dict = [rec.model_dump() for rec in processed_records]
        df = pd.DataFrame(records_dict)

        # Save to Parquet with PyArrow engine
        df.to_parquet(dest_file, engine="pyarrow", index=False)
        logger.info(f"[SUCCESS] Successfully saved {len(df)} complaints to {dest_file}")

        if underfilled:
            logger.warning(
                f"[!] {len(underfilled)}/{len(cells)} cells did not reach quota: "
                + "; ".join(underfilled)
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
        default=settings.MAX_COMPLAINTS_TO_INGEST,
        help=f"Number of valid complaints to ingest (default: {settings.MAX_COMPLAINTS_TO_INGEST})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Records requested per API call (default: 100)",
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
        batch_size=args.batch_size,
        output_file=Path(args.output),
    )

    if df.empty:
        return

    family_of = settings.PRODUCT_TO_FAMILY
    df["product_family"] = df["product"].map(family_of).fillna("Other")
    cell_sizes = df.groupby(["company", "product_family"]).size()

    logger.info("\n[SUMMARY] Dataset Ingestion Summary:")
    logger.info(f"Total Rows: {len(df)}")
    logger.info(f"Unique Companies: {df['company'].nunique()}")
    logger.info(f"Product Families: {df['product_family'].nunique()}")
    logger.info(f"Cells Populated: {len(cell_sizes)} | Median Rows/Cell: {int(cell_sizes.median())}")
    logger.info(f"Monetary Relief Rate: {(df['has_monetary_relief'].mean() * 100):.1f}%")
    logger.info(f"Average Narrative Words: {df['word_count'].mean():.0f}")


if __name__ == "__main__":
    main()
