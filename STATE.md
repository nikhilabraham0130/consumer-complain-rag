# 📍 Project State & Agent Handoff

**Active Phase:** Phase 8 — API & Dashboard Serving  
**Current Milestone:** Phase 7 RAGAS Generation Quality Benchmark Completed (95.0% Faithfulness, 91.3% Relevance, 100% Citation Precision)  
**Last Updated By:** Antigravity (2026-09-06)  

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

- [x] **Phase 6 — Retrieval Ablation Benchmark Study:**
  - `evals/metrics.py`: reciprocal rank, recall at k, ndcg at k (with graded relevance). 12/12 unit tests passing.
  - `evals/ablation.py`: 5-stage ablation runner evaluating BM25, Dense BGE, Hybrid+RRF, Cross-Encoder, and MMR across all 45 answerable queries.
  - Empirical findings: Hybrid+RRF achieved 100.0% Recall@50, peak MRR@10 of 0.7708 (+43.8% lift over BM25), and 116.5ms p50 latency, proving superior cost/latency efficiency over CPU-bound cross-encoders.
  - Official reports saved to `evals/results/ablation_results.json` and `evals/results/ablation_results.md`.
  - Comprehensive guide in `.work/phase_6_deep_dive.md`.

---

## 2. In Progress / Immediate Next Steps

- [x] **Phase 7 — RAGAS Generation Quality Suite:**
  - `evals/ragas_schemas.py`: mathematical models and schemas for claim extraction, verification, and evaluation results.
  - `src/utils/dns_patch.py`: network resiliency patch for DNS routing.
  - `evals/ragas_eval.py`: full evaluation harness decomposing answers into atomic propositions audited by DeepSeek-V3 judge.
  - Official benchmark executed across all 45 answerable queries:
    - **Faithfulness (Factual Grounding):** 95.0% (PASS, >90% target)
    - **Answer Relevance:** 91.3% (PASS, >85% target)
    - **Citation Precision:** 100.0% (Deterministic guardrail)
    - **Hallucination Rate:** 5.0% (SAFE, <10% target)
  - Official reports saved to `evals/results/ragas_eval_results.json` and `evals/results/ragas_eval_results.md`.
  - Comprehensive student learning guide in `.work/phase_7_deep_dive.md`.

---

## 2. In Progress / Immediate Next Steps

- [ ] **Phase 8 — API & Dashboard Serving:**
  - [x] FastAPI production REST service (`api/main.py`):
    - `POST /query`: Grounded synthesis inference with telemetry & guardrails.
    - `GET /health`: Corpus & index readiness/liveness probe.
    - `GET /metrics`: Automated telemetry exposing Phase 6 & Phase 7 benchmarks.
    - `GET /benchmark/queries`: Catalog of 50 golden benchmark queries.
    - Automated unit & integration tests (`tests/test_api.py`): 7/7 tests passing (95/95 total test suite passing).
  - [ ] Streamlit compliance UI (`ui/app.py`): multi-bank search, citation verification badges, Wilson score relief probability charts, and telemetry breakdowns.
- [ ] **Phase 9 — Production Packaging & Documentation:**
  - Docker containerization (`Dockerfile`, `docker-compose.yml`).
  - Production README with benchmark tables and architecture diagrams.

---

## 3. Core Architectural Invariants (Do Not Change)

- **Pipeline order:** `RRF → Cross-Encoder (top-50) → MMR (top-5)`.
- **CFPB ingestion:** Always use `search_after` cursor pagination.
- **Models:** `BAAI/bge-small-en-v1.5` (embeddings) and `BAAI/bge-reranker-base` (reranking).
- **LLM:** DeepSeek-V3 via OpenAI-compatible endpoint with automatic Gemini fallback.
- **Git:** All git commits and pushes are executed manually by the user.
