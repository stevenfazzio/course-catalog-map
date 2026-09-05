# Embedding model comparison

Generated 2026-09-05T07:07:09+00:00 by `pipeline/compare_embeddings.py` per `docs/preregistration.md`. Corpus: corpus.parquet (11,091 courses).

## Decision

**qwen3-0.6b**: no candidate beats the incumbent (step 4)

## Primary metric: embedding space, k = 15, breadth category (559 labelled courses; shuffled-label floor 0.306)

| model | agreement | diff vs incumbent | 95% paired CI |
|---|---|---|---|
| qwen3-0.6b (incumbent) | 0.815 | | |
| qwen3-4b | 0.814 | -0.001 | [-0.009, +0.008] |
| arctic-l-v2 | 0.733 | -0.082 | [-0.094, -0.070] |
| nomic-v2-moe | 0.746 | -0.069 | [-0.081, -0.057] |

## All labels, embedding space (rows: model; floor = shuffled labels)

| model | breadth (GER:DB) k=15 | department k=15 | school k=15 | WAYS k=15 | breadth (GER:DB) k=50 | department k=50 | school k=50 | WAYS k=50 |
|---|---|---|---|---|---|---|---|---|
| floor | 0.306 | 0.015 | 0.300 | 0.300 |  |  |  |  |
| qwen3-0.6b | 0.815 | 0.428 | 0.712 | 0.806 | 0.739 | 0.341 | 0.675 | 0.760 |
| qwen3-4b | 0.814 | 0.484 | 0.747 | 0.816 | 0.743 | 0.398 | 0.714 | 0.770 |
| arctic-l-v2 | 0.733 | 0.373 | 0.685 | 0.754 | 0.634 | 0.279 | 0.632 | 0.696 |
| nomic-v2-moe | 0.746 | 0.372 | 0.684 | 0.759 | 0.655 | 0.282 | 0.634 | 0.709 |

## 2-d layout, k = 15, mean [min, max] over UMAP seeds 42/43/44; trustworthiness of the layout

| model | breadth (GER:DB) | department | school | WAYS | trustworthiness |
|---|---|---|---|---|---|
| qwen3-0.6b | 0.772 [0.759, 0.782] | 0.387 [0.386, 0.388] | 0.688 [0.686, 0.690] | 0.765 [0.762, 0.767] | 0.989 [0.989, 0.990] |
| qwen3-4b | 0.780 [0.767, 0.792] | 0.444 [0.443, 0.444] | 0.714 [0.713, 0.716] | 0.778 [0.774, 0.782] | 0.991 [0.990, 0.991] |
| arctic-l-v2 | 0.748 [0.737, 0.760] | 0.305 [0.305, 0.307] | 0.639 [0.637, 0.641] | 0.719 [0.715, 0.724] | 0.930 [0.929, 0.932] |
| nomic-v2-moe | 0.746 [0.732, 0.758] | 0.316 [0.313, 0.318] | 0.648 [0.645, 0.651] | 0.709 [0.707, 0.709] | 0.952 [0.951, 0.952] |

## Neighbourhood overlap between models (mean Jaccard of 15-NN sets)

| pair | embedding space | 2-d (seed 42) |
|---|---|---|
| qwen3-0.6b vs qwen3-4b | 0.443 | 0.252 |
| qwen3-0.6b vs arctic-l-v2 | 0.280 | 0.134 |
| qwen3-0.6b vs nomic-v2-moe | 0.309 | 0.157 |
| qwen3-4b vs arctic-l-v2 | 0.257 | 0.127 |
| qwen3-4b vs nomic-v2-moe | 0.276 | 0.144 |
| arctic-l-v2 vs nomic-v2-moe | 0.482 | 0.181 |
