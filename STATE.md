# 📍 Project State & Agent Handoff

**Active Phase:** Phase 4 — Golden Evaluation Benchmark (Transitioning to Phase 5 & 6)  
**Current Milestone:** Golden set construction complete (1,825 labeled judgements across 50 queries)  
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
  - `src/retrieval/sparse_index.py`: BM25 via `rank_bm25` with regex citation carve-out (`§\d[\d.]*`).
  - `src/retrieval/dense_index.py`: ChromaDB persistent index with BGE embeddings (`BAAI/bge-small-en-v1.5`), query instruction prefix, metadata preservation, and 2,000-row batch chunking.
  - `src/retrieval/hybrid_search.py`: RRF fusion (`k=60`) → Cross-Encoder reranking (`BAAI/bge-reranker-base`) → MMR diversity selection (`lambda=0.6`).
  - Comprehensive unit tests: 57/57 tests passing.
- [x] **Phase 4 — Golden Evaluation Set Construction:**
  - `evals/queries.py`: 50 evaluation queries (40 stratified across 5 banks × 4 products + 10 cross-bank queries).
  - `evals/build_pools.py`: TREC-style pooling (union of top-20 BM25 and top-20 Dense).
  - `evals/golden_set.jsonl`: Master dataset of 1,825 candidate judgements, 100% labeled (`relevant`, `partially_relevant`, `not_relevant`) with explicit reasoning.

---

## 2. In Progress / Immediate Next Steps

- [ ] **Phase 4 Scaffolding Completion:**
  - `evals/golden_set.py`: Clean dataset loader, label mapping (`relevant: 2`, `partially_relevant: 1`, `not_relevant: 0`), and schema verification.
  - `tests/test_golden_set.py`: Pytest verification for golden set integrity.
- [ ] **Phase 5 — Self-Querying & Grounded Generation:**
  - `src/rag_pipeline.py`: Pydantic self-query filter extraction, grounded prompt synthesis with Complaint ID citations, and empirical resolution predictor with 95% CI.
- [ ] **Phase 6 — Retrieval Ablation Study:**
  - `evals/ablation.py`: Run all 50 queries against 5 configurations (BM25 only, Dense only, Hybrid+RRF, +Cross-Encoder, +MMR) and measure Recall@50, MRR@10, nDCG@10, and p95 latency.

---

## 3. Core Architectural Invariants (Do Not Change)

- **Pipeline order:** `RRF → Cross-Encoder (scores top-50) → MMR (selects final top-5)`. Whichever runs last determines what the caller sees.
- **CFPB ingestion:** Always use `search_after` cursor pagination; `frm` is ignored by the API.
- **Models:** `BAAI/bge-small-en-v1.5` (embeddings) and `BAAI/bge-reranker-base` (reranking).
- **ChromaDB upserts:** Always chunk upserts (`_UPSERT_CHUNK_SIZE = 2000`) to respect ChromaDB's max batch size limit (5,461).
- **Git:** All git commits and pushes are executed manually by the user.
