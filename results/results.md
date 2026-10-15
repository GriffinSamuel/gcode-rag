| config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- |
| dense_baseline | 0.6893 | 0.9435 | 0.7880 |
| bm25_only | 0.6215 | 0.8757 | 0.7314 |
| hybrid_rrf | 0.7401 | 0.9435 | 0.8278 |

- Questions: 177; embedding model: BAAI/bge-small-en-v1.5; corpus: 150 docs / 276 chunks (sources: {'linuxcnc': 150, 'haas': 0})
- Best config by recall@5: hybrid_rrf

| split | n | config | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| dev | 71 | dense_baseline | 0.6479 | 0.9577 | 0.7697 |
| dev | 71 | bm25_only | 0.6197 | 0.8732 | 0.7304 |
| dev | 71 | hybrid_rrf | 0.7324 | 0.9296 | 0.8176 |
| test | 106 | dense_baseline | 0.7170 | 0.9340 | 0.8002 |
| test | 106 | bm25_only | 0.6226 | 0.8774 | 0.7321 |
| test | 106 | hybrid_rrf | 0.7453 | 0.9528 | 0.8347 |

- Answer evaluation (openai_compatible qwen2.5-coder:7b, hybrid_rrf): keyword accuracy 0.8757 over 177 questions, citation rate 0.8927
  - dev: keyword accuracy 0.8310, citation rate 0.8732 (n=71)
  - test: keyword accuracy 0.9057, citation rate 0.9057 (n=106)
