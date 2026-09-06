"""
Pydantic Schemas for RAGAS Generation Quality Evaluation.

Defines the mathematical contracts for:
1. Atomic Claim Extraction (decomposing LLM response into verifiable propositions)
2. Claim Grounding Verification (verifying each claim strictly against retrieved context)
3. Answer Relevance Evaluation (measuring query-response alignment)
4. Overall RAGAS evaluation reports and telemetry
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ClaimExtractionResponse(BaseModel):
    """Structured decomposition of an executive summary into atomic factual claims."""
    claims: List[str] = Field(
        default_factory=list,
        description="List of independent, atomic factual statements extracted from the text."
    )

    @field_validator("claims", mode="before")
    @classmethod
    def clean_claims(cls, v: list) -> List[str]:
        if not v:
            return []
        cleaned = []
        for c in v:
            if isinstance(c, str) and c.strip():
                cleaned.append(c.strip())
        return cleaned


class AtomicClaimVerdict(BaseModel):
    """Grounding verification verdict for a single atomic claim."""
    claim: str = Field(description="The atomic claim evaluated.")
    is_grounded: bool = Field(
        description="True if the claim is directly supported by the retrieved context; False if hallucinated or unsubstantiated."
    )
    supporting_complaint_ids: List[str] = Field(
        default_factory=list,
        description="Complaint IDs from the context providing direct evidence for this claim."
    )
    reasoning: str = Field(
        description="Brief justification explaining why the claim is supported or unsupported."
    )


class ClaimVerificationResponse(BaseModel):
    """Batch verification judgments for extracted claims against retrieved context."""
    verdicts: List[AtomicClaimVerdict] = Field(
        default_factory=list,
        description="Verification verdict for each extracted atomic claim."
    )


class AnswerRelevanceResponse(BaseModel):
    """Evaluation of how directly and completely the generated summary answers the query."""
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score between 0.0 (completely irrelevant) and 1.0 (perfectly addresses all facets of the query)."
    )
    is_complete: bool = Field(
        description="True if all aspects of the user inquiry are answered."
    )
    reasoning: str = Field(
        description="Explanation of the score and any missing aspects."
    )


class QueryRagasResult(BaseModel):
    """End-to-end generation evaluation metrics for a single query."""
    query_id: str
    query: str
    target_company: Optional[str] = None
    product_family: Optional[str] = None
    executive_summary: str
    total_claims: int
    grounded_claims: int
    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="Ratio of grounded claims to total claims (1.0 = zero hallucinations)."
    )
    answer_relevance: float = Field(
        ge=0.0,
        le=1.0,
        description="Semantic relevance and completeness score (0.0 - 1.0)."
    )
    citation_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="Deterministic citation accuracy verified by CitationGuardrail."
    )
    claims_detail: List[AtomicClaimVerdict] = Field(default_factory=list)


class RagasBenchmarkSummary(BaseModel):
    """Aggregated generation evaluation summary across all benchmark queries."""
    total_queries: int
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_citation_precision: float
    hallucination_rate: float = Field(
        description="1.0 - mean_faithfulness (percentage of claims unsupported by context)."
    )
