# Retrieval Ablation Study Benchmark Results

Empirical performance comparison across 5 retrieval architectures on the CFPB Consumer Complaint Intelligence dataset.

| Retrieval Configuration | Recall@50 | MRR@10 | nDCG@10 | MRR Lift vs BM25 | p50 Latency | p95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 Only** | 64.1% | 0.5359 | 0.4291 | 0.0% | 56.5ms | 94.0ms |
| **Dense BGE Only** | 82.9% | 0.7347 | 0.6657 | +37.1% | 59.6ms | 166.8ms |
| **Hybrid + RRF** | 100.0% | 0.7708 | 0.6278 | +43.8% | 116.5ms | 224.3ms |
| **Hybrid + Cross-Encoder** | 100.0% | 0.6452 | 0.5424 | +20.4% | 21404.8ms | 29438.2ms |
| **Full Pipeline (+ MMR)** | 100.0% | 0.6552 | 0.5395 | +22.3% | 21557.8ms | 29653.8ms |

> **Evaluation Protocol:** Measured across 45 answerable compliance queries from the frozen `golden_set.jsonl` benchmark (TREC-style pooled judging across 1,824 candidate dockets).