# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

An interactive datamap of every course in a university's catalog: embed each course's title and
description, lay the corpus out in 2-d with UMAP, name the regions with Toponymy, render with
DataMapPlot. The first map is Stanford (ExploreCourses, academic year 2026-27). More universities
may follow; each is an adapter under `pipeline/sources/` that normalises to one common schema, and
everything downstream of stage 01 is source-agnostic.

**Audience and standards.** The intended audience skews more academic than Steven's other maps
(Steam games, Jeopardy). Err on the side of rigor and transparency: pinned versions and seeds,
provenance recorded (snapshot date, catalog year, model revisions), every filter and collapse rule
documented with its count, LLM-generated region names disclosed as such. Not journal-grade, but
nothing an academic reader would call misleading or sloppy.

## Running the pipeline

```bash
uv sync --extra dev
uv run python pipeline/00_fetch.py --source stanford        # ExploreCourses XML per department -> data/stanford/raw/
uv run python pipeline/01_parse.py --source stanford        # raw XML -> listings.parquet, courses.parquet, drops.csv
uv run python pipeline/01b_clean.py --source stanford       # preregistered corpus rules -> corpus.parquet (+ embed_text)
uv run python pipeline/02_embed.py --source stanford        # embeds corpus.parquet embed_text; --device runpod for a GPU pod
uv run python pipeline/03_reduce_umap.py --source stanford  # UMAP 1024-d -> 2-d, fixed seed -> umap_coords.npz
uv run python pipeline/04_label_topics.py --source stanford # Toponymy + Claude region names -> labels.parquet
uv run python pipeline/05_visualize.py --source stanford    # DataMapPlot -> data/stanford/stanford_course_map.html
```

`make fetch|parse|embed|umap|label|visualize|map` wrap the same commands (`map` runs 02-05); `make lint`,
`make test`. Useful flags: `02_embed.py --limit 200` measures throughput without writing;
`04_label_topics.py --sweep` reports region counts per layer at several granularities with no LLM calls.
Stage 04 is the only stage that costs money (roughly ten dollars of Claude Opus 5 per map). Run it the way
`make label` does: `OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1`.
On macOS torch, scikit-learn and numba each load their own `libomp.dylib` into the process and the
multi-runtime OpenMP deadlocked stage 04 three times in a row (main thread parked in `__kmp_join_call`,
0% CPU, no error); a single OpenMP thread sidesteps it and costs little because embedding runs on MPS.
`HF_HUB_OFFLINE` skips a live Hub check that is unnecessary once stage 02 has cached the model. Do not
pipe the stage through a plain `grep`; it buffers everything until exit.

**Embedding models and corpora.** `config.EMBED_MODELS` is a registry keyed by short name; `EMBED_MODEL_KEY`
is the map's model. Stages 02-05 take `--model` / `--embedding <key>` and `--raw`; the default key on the
cleaned corpus owns the unsuffixed artifact names, any other key writes `_<key>`-suffixed copies, and
`--raw` (the 11,500-course catalog before the stage-01b rules, embedded as title + description) adds
`_raw` (`config.keyed_files`). The two exploration maps are the `_raw` builds of `qwen3-0.6b` and
`arctic-l-v2`; the preregistered sensitivity row runs on `_raw` embeddings of every candidate.

**Runpod.** `02_embed.py --device runpod` and `04_label_topics.py --device runpod` run the same scripts on
a throwaway GPU pod via `pipeline/remote.py` (runpodctl + ssh) and pull the results back: ~136 courses/s
for embedding versus ~7/s on the M3, and a whole stage-04 sweep round trip in ~2 minutes. The runner reads
per-data-center stock and walks `RUNPOD_GPU_CANDIDATES` in order, because an unconstrained create can be
"rented" in a site with no machine and never get an ssh port (two such orphans on 2026-09-05; they are now
deleted after a 4-minute wait and the next placement tried). The Anthropic key reaches the pod as a
remote `.env`, never on a command line. The pod is deleted in a `finally` unless `--keep-pod`; this
runpodctl build has no terminate-after guard, so check `runpodctl pod list` after any crash. Needs
`RUNPOD_API_KEY` or `~/.runpod/config.toml` and the account's registered SSH key at `~/.ssh/id_ed25519`.
Cluster extraction is not bit-identical across machines: the same layout gave 378 finest regions locally
and 381 on the pod, so a map's labels belong to the machine that produced them; the labels files are the
record, not something to regenerate.

**Preregistration.** `docs/preregistration.md` fixes how the embedding model is chosen (kNN label
agreement against the registrar's breadth categories, paired bootstrap, keep-the-incumbent default).
Part 1 is the evaluation design; part 2 will freeze the corpus after a declared exploratory phase on
the incumbent's and arctic's maps. Do not compute comparison metrics before part 2 is committed.
`pipeline/compare_embeddings.py` implements part 1 and is tested on synthetic data only.
`experiments/explore_regions.py --embedding <key>` is the exploratory-phase helper: it describes each
named region by content and metadata (dominant departments, boilerplate description share, title
patterns, unscheduled share) so that hygiene rules can be written against the data, never against
map position. Exploration maps are served with `python3 -m http.server 8765 --bind 127.0.0.1` from
`data/stanford/`; never open them via `file://`. Stage scripts are run
from the repo root; they import sibling modules (`config`, `io_utils`, `sources`) because
`pipeline/` is `sys.path[0]` when a script there is executed. Tests get the same path via
`conftest.py`.

`01_parse.py --only CS,HISTORY` parses a subset and writes nothing, for iterating on parsing rules.

## Data layout

`data/` is gitignored. Per source:

```
data/stanford/raw/_departments.xml   school -> department index
data/stanford/raw/<DEPT>.xml         one search result per department, sections inline (217 MB total)
data/stanford/raw/_manifest.json     academic year, fetch timestamp, filters used
data/stanford/listings.parquet       one row per (course, code) listing, before collapse
data/stanford/courses.parquet        one row per course, common schema (sources/__init__.py)
data/stanford/drops.csv              every listing removed by the collapse, with the rule and the row kept
data/stanford/embeddings.npz         course_id + unit-norm float32 embeddings; embeddings_meta.json records model revision etc.
data/stanford/umap_coords.npz        course_id + 2-d layout; umap_meta.json records the parameters
data/stanford/labels.parquet         course_id + cluster_layer_i (int, -1 = unlabelled) + label_layer_i (name); layer 0 is FINEST
data/stanford/topic_names.json       region names per layer; cluster_tree.json the parent edges; labels_meta.json the run record
data/stanford/stanford_course_map.html
```

Every `*.npz` is aligned to `courses.parquet` row order and carries `course_id`; the readers assert it.

Raw files are the provenance record: never delete or regenerate them casually. Stage 00 skips any
department file that already exists; delete a file to refetch it.

## Conventions

- Data files are written via `io_utils.write_parquet_safely` / `atomic_write_bytes` (tmp + verify +
  `os.replace`). Never write directly to a live data path.
- Every stage prints its row delta with a reason. An unexplained drop is a bug.
- `uv` for the environment, `ruff` for lint and format (line length 120, isort with local modules as
  first-party). Python 3.12 (`.python-version`).
- Fixed `random_state` on UMAP; `cvd_safer=True` and glasbey palettes in DataMapPlot; search on a
  composed field; hovercard with code, title, school, units, and a click-through to ExploreCourses.
  No histogram, no topic tree. See `~/.claude/skills/datamap` for the full defaults.

## Decisions so far (2026-09-04)

- **Stanford first, MIT second.** Chosen on data availability, size, and prestige. Probed and
  workable: MIT FireRoad (one JSON call, 7,168 courses, already validated), Cornell class roster API,
  UIUC Course Explorer. Berkeley, Harvard, Princeton, CMU block plain HTTP and would need browser
  scraping.
- **Embedding text = title + description only.** Department and school stay out of the text so the
  layout is content-driven and the org-structure colormaps can disagree with it. The cross-listing
  suffix ExploreCourses appends to titles, e.g. "(LINGUIST 284, SYMSYS 195N)", is stripped from
  `title` (kept in `title_raw`) for the same reason.
- **Embedding model: Qwen/Qwen3-Embedding-0.6B** (Apache 2.0), pinned to a Hugging Face revision hash in
  `config.py`, fp32, unit-normalised. Confirmed by the preregistered comparison on 2026-09-05
  (`docs/embedding_comparison.md`): the 4B sibling tied it on the primary metric and led on the
  secondary ones, which the rule does not act on; the two encoder-family candidates were clearly behind. Every text is embedded with the instruction "Identify the
  topic or theme of the given university course description", the form the Qwen3 Embedding report uses
  for clustering tasks. Toponymy's keyphrases and exemplars go through the same wrapper (`embedder.py`)
  so they share the space. A 4B comparison run is on the table before the final build.
- **Clustering happens in the 2-d layout.** One UMAP (n_neighbors 15, min_dist 0.05, cosine, seed 42)
  feeds both the plot and Toponymy's `clusterable_vectors`, so named regions match what a viewer sees.
- **Region names come from Claude Opus 5** via `LoopSafeNamer` in `04_label_topics.py`, a subclass of
  Toponymy's `AsyncLiteLLMNamer` built the way `AsyncAnthropicNamer` builds it, with two changes.
  `provider_kwargs={"drop_params": True}`: Opus 5 rejects the `temperature` parameter Toponymy sends,
  and litellm drops it, so naming runs at the model default of 1.0. And one semaphore per event loop:
  Toponymy 0.5.4 runs each naming pass in its own `asyncio.run()`, so the namer's single semaphore binds
  to the first loop and every contended call afterwards fails with "bound to a different event loop"
  (70 exhausted-retry failures on the first arctic run). Worth an upstream fix; Steven contributes to
  Toponymy. `NAMER_STYLE` caps names at five words; Opus 5 otherwise wrote 11-word finest-layer names.
  Names are LLM-generated labels and the methodology page must say so. An open naming model on Runpod
  is a possible later experiment.
- **Colormaps in v1:** school, level (career), scheduled in 2026-27, primary section format, WAYS
  general-education requirement, max units. Department (254 values) is on hover and in search only.
- **No LLM enrichment in v1.** Descriptions are already summaries and the catalog metadata carries
  the colormaps. Not ruled out later; this corpus could illustrate the LLM-normalisation technique.
- **One point per registrar courseId.** A cross-listed course appears once in every department that
  lists it, with the same `courseId`. Copies are collapsed; all codes are kept in `codes`. The
  primary code is the listing in a department owned by the course's `academicOrganization` (mapping
  learned from single-listing courses), then lowest `offerNumber`, then alphabetical. 2,507 of 2,509
  cross-listed courses resolve by home department.
- **Department index aliases.** The index lists TAPS and ILAC twice under different long names; each
  is treated as one department and the first long name is kept.
- **Corpus rules (stage 01b, preregistration part 2, 2026-09-05).** Developed in a declared
  exploratory phase on the two raw-catalog maps, written against content and metadata only, applied in
  order, every drop logged to `corpus_drops.csv`. Rule 1: descriptions under 3 words are placeholders
  (181 dropped). Rule 2: a whole description shared verbatim by 5+ courses across 5+ departments is
  administrative boilerplate (228 dropped in 12 groups: consent-of-instructor, repeat-for-credit,
  faculty-sponsored research units, Medical Scholars placements, doctoral practica, TGR units). Rule 3: a
  shared opening of 10+ words among 5+ courses is stripped from the embedded text, title alone if nothing
  remains (148 stripped, 91 title-only: PWR programme paragraphs, music lessons, Frosh 101, EE CPT, Law
  exchange programmes). 11,091 courses kept. Unscheduled courses (41%) are kept on purpose: the map is a
  map of what the university teaches, and the scheduled colormap shows the difference.

## Current state (2026-09-05)

The Stanford map is built on the cleaned corpus with the preregistered model: `data/stanford/stanford_course_map.html`,
11,091 courses, 357 / 120 / 42 / 14 named regions (naming ran on a Runpod pod in 5.6 minutes end to end).
Two raw-catalog exploration builds remain as `_raw` artifacts. `docs/index.html` is the map, `docs/record/`
its versioned record; v1 is "published but not promoted" as of 2026-09-05: public repo github.com/stevenfazzio/course-catalog-map,
Pages from `docs/`, live at https://stevenfazzio.com/course-catalog-map/ (the user site's custom domain;
the github.io address also serves it). The hosted copy returned the full 7.6 MB map and the record files. Methodology writing
moves to a blog post, not a docs page. **What remains before v1.1 is called complete is `docs/plan.md`**
(2026-09-05): eight ordered steps with the cut rules, the deferred items, and the decisions taken while planning;
strike items there as they land. The cosine guardrail ran clean (stage 06, `docs/record/stanford/name_guardrail.csv`).
`docs/lit_review.md` (2026-09-05) is the literature review, with candidate colormaps, validation targets, and
record metrics in its section 4; the post-hoc structure analysis lives in `experiments/`
(`evoc_layers.py` -> `name_evoc.py` [Runpod, Opus names for two EVoC layers per model] -> `structure_numbers.py`
-> `structure_report.py` -> `docs/embedding_structure.html`) and is not part of the preregistered comparison.
Serve `docs/` with `python3 -m http.server 8766 --bind 127.0.0.1` to view the report; it loads plotly from
cdn.plot.ly. Headline findings: family is the main axis of disagreement and it is present in the native
spaces (within-family AMI 0.80-0.89 vs 0.70-0.73 across; department-centroid Mantel 0.96 vs 0.72-0.80);
the encoder spaces have less density structure (EVoC leaves 37-46% unclustered vs 28-32% for Qwen) and lose
more in 2-d; 56 of 140 departments change nearest department between families.

**Plan step 1 landed (2026-09-05).** `experiments/record_metrics.py` (run with `OMP_NUM_THREADS=1`, 3.7 minutes,
no LLM; metric functions tested on synthetic data) writes `layout_metrics.json`, `stability.parquet` (per course, aligned
to the corpus), `stability_regions.csv`, `stability_meta.json` and `stability_layouts.npz` (every rerun's coordinates
and cluster layers) to `data/stanford/`, and stage 07 copies them into the record. The reference for stability is the
published `labels.parquet`, never a local re-clustering: the clusterer is deterministic on one machine but not across
machines (local re-clustering of the published layout gives 354 / 119 / 41 / 14 regions against the published
357 / 120 / 42 / 14, AMI 0.99+), and that control is in the record as the machine floor. Published layout: KNN 0.45
(k = 15), KNC 0.45 over 237 departments, 0.54 over the 121 with 20+ courses, 0.67 over schools, CPD 0.49, trustworthiness
0.99. At `n_neighbors` 100: KNN 0.34, CPD 0.53, KNC unchanged (reported, not adopted). Ten UMAP seeds: AMI against the
published layers 0.92 / 0.94 / 0.91 / 0.86 finest to coarsest, 14-16% of published members left unclustered; ten 80%
subsamples 0.89 / 0.91 / 0.88 / 0.82. Median region co-assignment 0.68 at the finest layer (71 of 357 regions above
0.9, 10 below 0.2); the coarsest layer's large regions split (median 0.60). Per course, `stay_seeds_layer_i` is the
share of its published-region peers it stays with, NaN where the published layer leaves it unlabelled: the step-6
hover or colormap input. The seed layouts in the npz were meant as step 2's null layouts; step 2 turned out not
to need them, so they remain for step 6.

**Plan steps 2 and 3 (2026-09-06).** Both are name evaluations and share `experiments/published_map.py` (the map's
record as one object: corpus text, embeddings, layout, labels, names, tree) and `experiments/listener.py` (Claude
Sonnet 5 through the Anthropic SDK with structured output, eight threads, a JSONL resume cache keyed by question id
with per-call token usage, refusals recorded rather than raised). Sonnet 5 rather than the namer, Opus 5, is the
no-self-preference rule from Steven's Toponymy evaluation; Sonnet 5 takes no temperature parameter, so repeats
measure the band at the model default. Step 2, `experiments/wayfinding_lineup.py` (`--dry-run` builds the items
and estimates the calls without spending; `--quick` is a twenty-call smoke test): probability mass on the true
region 0.58 [0.55, 0.60] over 135 items against chance 0.20, floor 0.18, ceiling 0.93, top-1 0.93, repeat band
p90 0.12; one item ("Emerging Viral Diseases And Biosecurity") dropped because the listener's safety classifier
refused it, category bio; about $5.50 for 680 calls. Outputs `wayfinding_items.json`, `wayfinding_lineup.csv`,
`wayfinding_lineup.json`, copied into the record by stage 07. Step 3, `experiments/name_intrusion.py`: fifty
items written to `name_intrusion_items.csv`, a sealed `name_intrusion_key.json`, and a rater page
`name_intrusion_rater.html` (serve `data/stanford/` on 8765; answers are kept in localStorage and copied out as
CSV); `--llm` had the listener answer them (48/50) and `--score` writes `name_intrusion.json`. Steven rated all
fifty on 2026-09-06: 45/50 (Wilson 95% [0.79, 0.96]), same pick as the listener on 45, both wrong on two; three of
his five misses had an intruder from a content-adjacent top-level region, which the tree-based intruder rule does
not exclude. Both instruments are unit-tested on synthetic maps in `tests/`. Run everything with
`OMP_NUM_THREADS=1` as for stage 04.

## Stanford data facts (2026-27 catalog, fetched 2026-09-05 UTC)

- 254 departments in 9 schools, 16 with zero active courses.
- 15,649 listings -> 11,500 courses. Largest departments by listings: HISTORY 754, LAW 690, ENGLISH
  432, EDUC 395, CEE 369, MUSIC 365, CS 363.
- By school: Humanities & Sciences 5,805; Engineering 1,557; Medicine 1,163; VPUE 1,017; Law 697;
  Sustainability 445; GSB 426; Education 307; Athletics 83.
- Career: UG 6,147; GR 3,969; LAW 691; GSB 403; MED 290.
- Description length: median 88 words, p10 22, p90 211.
- Colormap candidates in the metadata: school, career, `scheduled`, components (LEC/SEM/LAB/INS...),
  WAYS general-education tags (`gers`), units. Department (254 values) belongs on hover and search.

## Stanford ExploreCourses API notes

- Department index: `GET /?view=xml-20200810`.
- Per department: `GET /search?view=xml-20200810&academicYear=20262027&filter-coursestatus-Active=on
  &filter-departmentcode-CODE=on&q=CODE`. Sections are ~93% of the bytes.
- **Click-through (2026-09-05).** Stanford Navigator's Navigate Classes (April 2024) is the official catalog
  and is slated to replace ExploreCourses, which Stanford now calls the heritage catalog and which returned
  503 for hours after our fetch. Scheduled courses link to Navigator's exact class page,
  `navigator.stanford.edu/classes/<termId>/<classId>`, built from the section ids in the XML (verified:
  CS 329X Autumn 2026 -> /classes/1272/27853); 6,257 of 11,091 courses. Unscheduled courses have no
  Navigator page (it lists sections, not the catalog), so they keep the ExploreCourses catalog URL
  `/search?view=catalog&academicYear=20262027&q=CS224N&filter-departmentcode-CS=on&filter-coursestatus-Active=on`
  as a fallback that works only while that site is up (`url_catalog` keeps it for every course).
- **Source risk.** The adapter depends on the ExploreCourses XML API. Navigator has no documented API, is a
  term-scoped Algolia front end, and does not carry unscheduled catalog entries, so it cannot replace the
  fetch one for one. The 217 MB raw snapshot in `data/stanford/raw/` may be the last easy copy of the
  2026-27 catalog in this form; archive it as a release asset.
