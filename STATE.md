# 📍 Project State & Agent Handoff

**Active Phase:** Phase 3 — Hybrid Retrieval & Reranking Engine
**Current Milestone:** Not started — scaffolding only (config + deps)
**Last Updated By:** Claude Code (2026-09-01)

Phase numbers below are fixed by `.work/deep_dive_plan.md` (Project 3 section) —
that file is the source of truth for phase numbering. Don't renumber locally;
if a phase list conflicts with that file, the file wins.

---

## 1. Completed So Far

- [x] **Phase 1 — Ingestion pipeline:**
  - `src/ingestion/cfpb_client.py`: cursor pagination (`search_after`), 20-cell balanced grid sampling
  - `src/schemas.py`: Pydantic models with `AliasChoices` mapping the CFPB API's actual field names
    (`complaint_what_happened` not `consumer_complaint_narrative`; `company_response` not
    `company_response_to_consumer`)
- [x] **Phase 2 — Balanced corpus:**
  - `data/processed/complaints_cleaned.parquet` — **8,831 rows, 0 duplicate complaint_ids,
    5 banks** (Wells Fargo, Chase, BofA, Capital One, Citi — no credit bureaus),
    17/20 company×product cells at full quota. Verified 2026-08-28.
- [x] **Phase 3 scaffolding (config only, no retrieval code yet):**
  - `pyproject.toml`: added `chromadb>=0.5.5`, `sentence-transformers>=3.0.0`, `rank_bm25>=0.2.2`
  - `src/config.py`: added `CHROMA_DB_DIR`, `CHROMA_COLLECTION_NAME`, `EMBEDDING_MODEL_NAME`
    (`BAAI/bge-small-en-v1.5`), `RERANKER_MODEL_NAME` (`BAAI/bge-reranker-base`),
    `RETRIEVAL_TOP_K=50`, `RRF_K=60`, `FINAL_TOP_N=5`, `MMR_LAMBDA=0.6`

## 2. Known Gaps (do this before writing retrieval code)

- [ ] **`requirements.txt` was never updated** — only `pyproject.toml` has the new deps. The two
  files have drifted; anyone running `pip install -r requirements.txt` won't get chromadb etc.
  Fix this first.

## 3. In Progress / Immediate Next Step

Full module-by-module spec: `.work/phase3_hybrid_retrieval_plan.md` — **read this before writing
any retrieval code**, it fixes a real ordering bug from an earlier draft (see §4).

- [ ] `src/retrieval/sparse_index.py` — BM25 via `rank_bm25`. **Standard tokenizer** (lowercase,
  strip punctuation, static stopword list), with **one** regex carve-out for `§\d[\d.]*` citations
  (e.g. `§1005.11`). Not a custom financial tokenizer — `cleaner.py` already collapses every dollar
  amount to the literal token `[AMOUNT]` in Phase 1, so there's nothing left to preserve there.
- [ ] `src/retrieval/dense_index.py` — ChromaDB, persisted to `data/chroma_db/`. Must store
  `company`, `product`, `product_family`, `date_received`, `has_monetary_relief`, `word_count` as
  metadata alongside each embedding (Phase 5 self-querying needs to filter on these later —
  embedding is the expensive step, don't make it redo-able).
- [ ] `src/retrieval/hybrid_search.py` — RRF fuse → cross-encoder rerank → MMR select final top-5.

## 4. Core Architectural Invariants (do not change without discussion)

- **Pipeline order:** `RRF → Cross-Encoder (scores top-50) → MMR (selects final top-5)`.
  MMR must run **last**, after the cross-encoder — not before it. If MMR runs first, the
  pure-relevance cross-encoder rerank right after it can resurface near-duplicates and undo the
  diversity fix. Whichever step runs last determines the output.
- **CFPB ingestion:** never use offset/`frm` paging — the API validates it but silently ignores it
  and repeats page 1. Always use `search_after` cursor pagination.
- **Embeddings:** `BAAI/bge-small-en-v1.5`. **Reranker:** `BAAI/bge-reranker-base`. Both local/CPU,
  $0 marginal cost.
- **No `nltk`** — a first-run `nltk.download('punkt')` network call is a silent trap. Plain regex
  tokenizer instead.
- **Git:** neither agent runs commits automatically — the user reviews and commits.

## 5. How to use this file

Before starting work: read this file, then `.work/deep_dive_plan.md` if the phase/milestone needs
more context. After finishing a chunk of work: update §1–3 here with what changed and what's next,
then ask the user to review and commit — `STATE.md` travels with the commit, so switching agents
mid-project means the next one reads accurate state instead of asking the user to re-explain it.
