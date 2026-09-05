# Embedding model comparison

Generated 2026-09-05T07:20:01+00:00 by `pipeline/compare_embeddings.py` per `docs/preregistration.md`. Corpus: courses.parquet (11,500 courses).

## Decision

**qwen3-0.6b**: no candidate beats the incumbent (step 4)

## Primary metric: embedding space, k = 15, breadth category (560 labelled courses; shuffled-label floor 0.306)

| model | agreement | diff vs incumbent | 95% paired CI |
|---|---|---|---|
| qwen3-0.6b (incumbent) | 0.815 | | |
| qwen3-4b | 0.815 | -0.001 | [-0.009, +0.007] |
| arctic-l-v2 | 0.732 | -0.083 | [-0.095, -0.071] |
| nomic-v2-moe | 0.746 | -0.070 | [-0.082, -0.058] |

## All labels, embedding space (rows: model; floor = shuffled labels)

| model | breadth (GER:DB) k=15 | department k=15 | school k=15 | WAYS k=15 | breadth (GER:DB) k=50 | department k=50 | school k=50 | WAYS k=50 |
|---|---|---|---|---|---|---|---|---|
| floor | 0.306 | 0.014 | 0.298 | 0.301 |  |  |  |  |
| qwen3-0.6b | 0.815 | 0.421 | 0.712 | 0.807 | 0.739 | 0.334 | 0.674 | 0.762 |
| qwen3-4b | 0.815 | 0.475 | 0.745 | 0.817 | 0.744 | 0.389 | 0.711 | 0.772 |
| arctic-l-v2 | 0.732 | 0.362 | 0.686 | 0.754 | 0.633 | 0.271 | 0.632 | 0.699 |
| nomic-v2-moe | 0.746 | 0.363 | 0.686 | 0.761 | 0.654 | 0.276 | 0.636 | 0.713 |

## 2-d layout, k = 15, mean [min, max] over UMAP seeds 42/43/44; trustworthiness of the layout

| model | breadth (GER:DB) | department | school | WAYS | trustworthiness |
|---|---|---|---|---|---|
| qwen3-0.6b | 0.778 [0.773, 0.788] | 0.379 [0.378, 0.380] | 0.690 [0.689, 0.691] | 0.767 [0.759, 0.772] | 0.989 [0.988, 0.989] |
| qwen3-4b | 0.763 [0.735, 0.788] | 0.434 [0.433, 0.435] | 0.714 [0.711, 0.717] | 0.774 [0.770, 0.780] | 0.991 [0.991, 0.991] |
| arctic-l-v2 | 0.745 [0.736, 0.758] | 0.299 [0.294, 0.302] | 0.643 [0.642, 0.644] | 0.721 [0.713, 0.728] | 0.935 [0.932, 0.939] |
| nomic-v2-moe | 0.731 [0.730, 0.734] | 0.306 [0.304, 0.306] | 0.652 [0.650, 0.653] | 0.712 [0.709, 0.715] | 0.954 [0.953, 0.955] |

## Neighbourhood overlap between models (mean Jaccard of 15-NN sets)

| pair | embedding space | 2-d (seed 42) |
|---|---|---|
| qwen3-0.6b vs qwen3-4b | 0.444 | 0.253 |
| qwen3-0.6b vs arctic-l-v2 | 0.280 | 0.136 |
| qwen3-0.6b vs nomic-v2-moe | 0.308 | 0.151 |
| qwen3-4b vs arctic-l-v2 | 0.258 | 0.132 |
| qwen3-4b vs nomic-v2-moe | 0.275 | 0.140 |
| arctic-l-v2 vs nomic-v2-moe | 0.483 | 0.191 |
