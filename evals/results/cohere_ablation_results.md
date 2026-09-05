# Enterprise Reranking Benchmark: Local vs. Cohere Cloud (45 Queries)

| Architecture | Recall@50 | MRR@10 | nDCG@10 | p50 Latency | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 Only** | 64.1% | 0.5359 | 0.4291 | 114.1ms | 132.2ms |
| **Dense BGE Only** | 82.9% | 0.7347 | 0.6671 | 85.0ms | 130.2ms |
| **Hybrid + RRF** | 100.0% | 0.7708 | 0.6278 | 205.4ms | 250.6ms |
| **Hybrid + Local Cross-Encoder** | 100.0% | 0.6452 | 0.5424 | 21404.8ms | 21404.8ms |
| **Hybrid + Cohere Rerank v3.5** | 100.0% | 0.7021 | 0.6461 | 513.8ms | 1093.8ms |