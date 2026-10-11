| config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| dense_baseline | 0.5714 | 0.9464 | 0.6961 |
| bm25_only | 0.5714 | 0.8571 | 0.7008 |
| hybrid_rrf | 0.7500 | 0.9464 | 0.8363 |

- Questions: 56; embedding model: BAAI/bge-small-en-v1.5; corpus: 150 docs / 276 chunks (sources: {'linuxcnc': 150, 'haas': 0})
- Best config by recall@5: hybrid_rrf
- Answer evaluation skipped: no LLM key configured
