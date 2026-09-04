"""
Application and Ingestion Settings.
Uses Pydantic Settings to load configuration from environment variables and .env files.
"""

from pathlib import Path
from typing import Dict, List, Optional
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
    
    # -------------------------------------------------------------------------
    # Corpus Design (see .work/deep_dive_plan.md Project 3, section 2)
    # -------------------------------------------------------------------------
    # The "Big 5" peer retail banks. Credit bureaus (Experian/TransUnion/Equifax)
    # are deliberately excluded: their narratives are ~2.3x more templated
    # (13.8% vs 6.1% near-duplicate pairs), they yield no monetary-relief
    # outcomes, and unfiltered they dominate ~78% of the CFPB stream.
    TARGET_COMPANIES: List[str] = [
        "WELLS FARGO & COMPANY",
        "JPMORGAN CHASE & CO.",
        "BANK OF AMERICA, NATIONAL ASSOCIATION",
        "CAPITAL ONE FINANCIAL CORPORATION",
        "CITIBANK, N.A.",
    ]

    # CFPB renamed product categories over time, so one logical product family
    # maps to several API values. Filtering on a single spelling silently drops
    # records (e.g. "Credit card" vs "Credit card or prepaid card").
    TARGET_PRODUCT_FAMILIES: Dict[str, List[str]] = {
        "Checking or savings account": [
            "Checking or savings account",
        ],
        "Credit card": [
            "Credit card",
            "Credit card or prepaid card",
            "Prepaid card",
        ],
        "Mortgage": [
            "Mortgage",
        ],
        "Debt collection": [
            "Debt collection",
        ],
    }

    # Date window for ingestion (CFPB publishes narratives on a lag)
    INGEST_DATE_MIN: str = "2022-01-01"
    INGEST_DATE_MAX: str = "2023-12-31"

    # Safety guard: max pages fetched per company x product cell. Prevents an
    # unfillable quota from paging through the entire 1.69M-row corpus.
    MAX_PAGES_PER_CELL: int = 60

    @property
    def ALL_TARGET_PRODUCTS(self) -> List[str]:
        """Flattened list of every CFPB product value across all target families."""
        return [v for values in self.TARGET_PRODUCT_FAMILIES.values() for v in values]

    @property
    def PRODUCT_TO_FAMILY(self) -> Dict[str, str]:
        """Reverse map from a raw CFPB product value to its normalized family name."""
        return {
            value: family
            for family, values in self.TARGET_PRODUCT_FAMILIES.items()
            for value in values
        }
    
    # -------------------------------------------------------------------------
    # Retrieval & Reranking (see .work/phase3_hybrid_retrieval_plan.md)
    # -------------------------------------------------------------------------
    CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "cfpb_complaints"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

    RETRIEVAL_TOP_K: int = 50   # candidates pulled from EACH of BM25 / dense before fusion
    RRF_K: int = 60             # standard RRF smoothing constant (Cormack et al.)
    FINAL_TOP_N: int = 5        # results returned to the caller after MMR
    MMR_LAMBDA: float = 0.6     # relevance vs. diversity tradeoff in MMR

    # -------------------------------------------------------------------------
    # LLM & Generation (Phase 5)
    # -------------------------------------------------------------------------
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT_SECONDS: int = 60

    # -------------------------------------------------------------------------


    # Logging & Environment
    # -------------------------------------------------------------------------
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    def ensure_directories_exist(self) -> None:
        """Helper to guarantee that data directories exist on disk."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)


# Global singleton instance for import across the codebase
settings = Settings()
settings.ensure_directories_exist()

