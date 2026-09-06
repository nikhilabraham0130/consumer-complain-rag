# RAGAS Generation Quality Benchmark Results (45 Queries)

Empirical assessment of hallucination rate, factual grounding, and answer relevance on CFPB compliance queries.

| Generation Metric | Measured Score | Industry Production Target | Audit Verdict |
| :--- | :---: | :---: | :---: |
| **Faithfulness (Factual Grounding)** | **95.0%** | > 90.0% | PASS |
| **Answer Relevance** | **91.3%** | > 85.0% | PASS |
| **Citation Precision (Guardrail)** | **100.0%** | 100.0% | PASS (Deterministic) |
| **Hallucination Rate** | **5.0%** | < 10.0% | SAFE (< 10%) |

> **Methodology:** Every generated summary was decomposed into atomic factual propositions using an independent LLM judge. Each claim was checked against the retrieved CFPB complaint context. Citations were verified deterministically against candidate IDs.