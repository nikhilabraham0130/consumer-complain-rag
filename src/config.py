"""
Application and Ingestion Settings.
Uses Pydantic Settings to load configuration from environment variables and .env files.
"""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for the Consumer Complaint Intelligence RAG system."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # -------------------------------------------------------------------------
    # Project Paths
    # -------------------------------------------------------------------------
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    # Cleaned parquet output path
    PROCESSED_PARQUET_FILE: Path = PROCESSED_DATA_DIR / "complaints_cleaned.parquet"
    
    # -------------------------------------------------------------------------
    # CFPB API Ingestion Settings
    # -------------------------------------------------------------------------
    CFPB_API_BASE_URL: str = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
    MAX_COMPLAINTS_TO_INGEST: int = 10000
    INGESTION_BATCH_SIZE: int = 250  # CFPB API max results per page is 100-250
    REQUEST_TIMEOUT_SECONDS: int = 30
    MAX_API_RETRIES: int = 4
    BACKOFF_FACTOR: float = 1.5
    
    # Target financial institutions for curated sampling
    TARGET_COMPANIES: List[str] = [
        "JPMORGAN CHASE & CO.",
        "WELLS FARGO & COMPANY",
        "BANK OF AMERICA, NATIONAL ASSOCIATION",
        "CITIBANK, N.A.",
        "CAPITAL ONE FINANCIAL CORPORATION",
        "EQUIFAX, INC.",
        "EXPERIAN INFORMATION SOLUTIONS INC.",
        "TRANSUNION INTERMEDIATE HOLDINGS, INC."
    ]
    
    # Target products
    TARGET_PRODUCTS: List[str] = [
        "Checking or savings account",
        "Credit card or prepaid card",
        "Mortgage",
        "Debt collection",
        "Credit reporting, credit repair services, or other personal consumer reports"
    ]
    
    # -------------------------------------------------------------------------
    # Logging & Environment
    # -------------------------------------------------------------------------
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    def ensure_directories_exist(self) -> None:
        """Helper to guarantee that data directories exist on disk."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Global singleton instance for import across the codebase
settings = Settings()
settings.ensure_directories_exist()

