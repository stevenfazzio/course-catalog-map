# course-catalog-map

An interactive semantic map of every course in a university catalog. Each point is a course,
placed by the meaning of its title and description, so courses that teach similar things land
near each other regardless of which department lists them.

First map: **Stanford**, academic year 2026-27, 11,500 courses from the public
[ExploreCourses](https://explorecourses.stanford.edu/) catalog.

## Pipeline

```
uv sync --extra dev
uv run python pipeline/00_fetch.py --source stanford        # raw catalog XML, one file per department
uv run python pipeline/01_parse.py --source stanford        # -> data/stanford/courses.parquet
uv run python pipeline/02_embed.py --source stanford        # Qwen3-Embedding-0.6B, local
uv run python pipeline/03_reduce_umap.py --source stanford  # UMAP to 2-d, fixed seed
uv run python pipeline/04_label_topics.py --source stanford # Toponymy region naming (Claude)
uv run python pipeline/05_visualize.py --source stanford    # DataMapPlot -> interactive HTML
```

Each university is an adapter in `pipeline/sources/` producing the common schema in
`pipeline/sources/__init__.py`; everything downstream is source-agnostic.

## Method, briefly

- One point per registrar course. Cross-listed copies are collapsed and every code is kept.
- Position comes from an open embedding model (Qwen3-Embedding-0.6B, pinned revision) applied to
  title + description only, reduced to 2-d with UMAP at a fixed seed.
- Regions are found by density clustering in the 2-d layout and named by an LLM (Claude Opus 5)
  through Toponymy. Names are labels for browsing, not ground truth.
- Nothing is filtered out: placeholder and boilerplate courses stay on the map so the map can show them.

See `CLAUDE.md` for the decision log, data facts, and provenance details.
