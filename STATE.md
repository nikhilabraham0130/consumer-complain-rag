# 📍 Project State & Agent Handoff

**Active Phase:** Phase 6 — Retrieval Ablation Benchmark Study  
**Current Milestone:** Phase 5 Generation Pipeline fully operational (7/7 tests passing, 100% citation precision verified live)  
**Last Updated By:** Antigravity (2026-09-04)  

Phase numbers below follow `.work/deep_dive_plan.md` (Project 3 section) — that file is the source of truth for phase numbering.

---

## 1. Completed So Far

- [x] **Phase 1 — Ingestion pipeline:**
  - `src/ingestion/cfpb_client.py`: cursor pagination (`search_after`), 20-cell balanced grid sampling.
  - `src/schemas.py`: Pydantic models with `AliasChoices` mapping CFPB API fields.
  - `src/ingestion/cleaner.py`: PII normalization and text scrubbing.
- [x] **Phase 2 — Balanced corpus:**
  - `data/processed/complaints_cleaned.parquet` — 8,831 rows, 0 duplicate complaint_ids, 5 banks, 17/20 company×product cells at full quota.
- [x] **Phase 3 — Hybrid Retrieval & Reranking Engine:**
  - `src/retrieval/sparse_index.py`: BM25 with regex citation carve-out (`§\d[\d.]*`).
  - `src/retrieval/dense_index.py`: ChromaDB persistent index with BGE embeddings (`BAAI/bge-small-en-v1.5`), query instruction prefix, and 2,000-row batch chunking.
  - `src/retrieval/hybrid_search.py`: RRF fusion (`k=60`) → Cross-Encoder reranking (`BAAI/bge-reranker-base`) → MMR diversity selection (`lambda=0.6`).
  - Comprehensive unit tests: 57/57 tests passing.
- [x] **Phase 4 — Golden Evaluation Set Construction:**
  - `evals/queries.py`: 50 evaluation queries (40 stratified + 10 cross-bank).
  - `evals/build_pools.py`: TREC-style pooling (union of top-20 BM25 and top-20 Dense).
  - `evals/golden_set.jsonl`: Master dataset of 1,824 candidate judgements, 100% labeled (`relevant`, `partially_relevant`, `not_relevant`).
  - `evals/golden_set.py`: Dataset loader, label mapping, and answerability partitions (45 active, 5 negative controls).
  - `tests/test_golden_set.py`: 6/6 tests passing.
- [x] **Phase 5 — Self-Querying & Grounded Generation Pipeline:**
  - `src/llm_client.py`: Multi-provider resilient client supporting DeepSeek-V3, Google Gemini, and OpenAI with schema prompt injection and exponential backoff.
  - `src/rag_pipeline.py`:
    - `SelfQueryExtractor`: Pydantic metadata filter extraction from natural language.
    - `CitationGuardrail`: Deterministic citation verification against retrieved context (verified live at 100.0% precision).
    - `ResolutionPredictor`: Wilson Score 95% Confidence Interval probability estimation.
    - `PipelineTelemetry`: Latency profiling across extraction, retrieval, reranking, and generation.
    - `ComplianceReport`: Structured executive report with automated regulatory risk scoring (`HIGH`, `MEDIUM`, `LOW`).
  - `tests/test_rag_pipeline.py`: 7/7 tests passing.

---

## 2. In Progress / Immediate Next Steps

- [ ] **Phase 6 — Retrieval Ablation Study (`evals/ablation.py`):**
  - Run all 45 answerable queries against 5 configurations:
    1. BM25 Only
    2. Dense BGE Only
    3. Hybrid + RRF
    4. Hybrid + RRF + Cross-Encoder
    5. Full Pipeline (+ MMR diversity)
  - Compute **Recall@50**, **MRR@10**, **nDCG@10**, and **p95 latency**.
  - Generate the publication-ready Markdown benchmark table for the portfolio/resume.
- [ ] **Phase 7 — RAGAS Generation Quality Suite (`evals/ragas_eval.py`):**
  - Faithfulness, Answer Relevance, Context Precision, and Context Recall.
- [ ] **Phase 8 — API & Dashboard Serving:**
  - FastAPI endpoints (`api/main.py`) and Streamlit compliance UI (`ui/app.py`).

---

## 3. Core Architectural Invariants (Do Not Change)

- **Pipeline order:** `RRF → Cross-Encoder (top-50) → MMR (top-5)`.
- **CFPB ingestion:** Always use `search_after` cursor pagination.
- **Models:** `BAAI/bge-small-en-v1.5` (embeddings) and `BAAI/bge-reranker-base` (reranking).
- **LLM:** DeepSeek-V3 via OpenAI-compatible endpoint with automatic Gemini fallback.
- **Git:** All git commits and pushes are executed manually by the user.
