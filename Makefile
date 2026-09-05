.PHONY: install lint format test fetch parse embed umap label visualize map clean

SOURCE ?= stanford

install:
	uv sync --extra dev

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

test:
	uv run pytest

fetch:
	uv run python pipeline/00_fetch.py --source $(SOURCE)

parse:
	uv run python pipeline/01_parse.py --source $(SOURCE)

embed:
	uv run python pipeline/02_embed.py --source $(SOURCE)

umap:
	uv run python pipeline/03_reduce_umap.py --source $(SOURCE)

# HF_HUB_OFFLINE: the model is cached by stage 02, and a live Hub check has hung this stage before.
# OMP_NUM_THREADS=1: torch, scikit-learn and numba each load their own libomp on macOS, and the
# resulting multi-runtime OpenMP deadlocked this stage three times; the heavy work is on the GPU anyway.
label:
	OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
		uv run python pipeline/04_label_topics.py --source $(SOURCE)

visualize:
	uv run python pipeline/05_visualize.py --source $(SOURCE)

map: embed umap label visualize

clean:
	@echo "This will remove all files in data/. Press Ctrl+C to cancel."
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/*
