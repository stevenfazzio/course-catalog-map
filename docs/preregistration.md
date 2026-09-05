# Preregistration: choosing the embedding model

**Status: parts 1 and 2 complete; committed before the comparison ran.** Part 1 fixes the evaluation
design. It was written on 2026-09-05 before any model-comparison metric was computed and before any map
had been viewed, and reviewed by Steven on 2026-09-05. Part 2 fixes the corpus and was written after the
declared exploratory phase (see "Exploratory phase" below), also on 2026-09-05. The comparison runs
once after both parts are committed. Deviations, if any, are recorded in the "Deviations" section at
the end rather than edited into the text above them.

## Question

Which open-weights embedding model should place the courses on the map? Operationalised: which
model's embedding space best recovers the university's own subject classification of courses,
given that the embedded text (title + description) contains no department, school, or requirement
metadata by construction.

## What was already known when part 1 was written

- The incumbent, Qwen3-Embedding-0.6B, had been embedded locally (fp32, MPS), laid out with UMAP,
  and run through a Toponymy granularity sweep that reports region counts only. Naming was in
  progress. No map had been rendered or viewed.
- Arctic-embed-l-v2.0 embeddings were being computed on a Runpod GPU at the time of writing. No
  metric of any kind had been computed on them.
- The evaluation labels below were chosen from catalog metadata before seeing any layout.

## Candidates (fixed)

| key | model | family | prompt applied to every text | max tokens |
|---|---|---|---|---|
| `qwen3-0.6b` (incumbent) | Qwen/Qwen3-Embedding-0.6B | Qwen3 decoder-only LLM, last-token pooling | "Instruct: Identify the topic or theme of the given university course description\nQuery: " | 1024 |
| `qwen3-4b` | Qwen/Qwen3-Embedding-4B | same family, larger | same | 1024 |
| `arctic-l-v2` | Snowflake/snowflake-arctic-embed-l-v2.0 | XLM-RoBERTa-large encoder, CLS pooling | none (document mode) | 1024 |
| `nomic-v2-moe` | nomic-ai/nomic-embed-text-v2-moe | NomicBERT mixture-of-experts encoder, mean pooling | "clustering: " | 512 (model maximum) |

All Apache-2.0. Revisions are pinned by commit hash in `pipeline/config.py`. Each model is used in
its documented clustering mode; prompts are frozen as listed and are not tuned after results.
Embeddings are fp32, unit-normalised, produced by sentence-transformers 5.7.0. A candidate that
fails to run (out of memory, remote-code failure) is dropped and noted, never substituted.

The incumbent's vectors are the local fp32 MPS run already produced, because those are the vectors
its map is built from. Candidates are embedded on a Runpod GPU (fp32, CUDA). A device-parity check
is reported: cosine similarity between MPS and CUDA embeddings of the incumbent on 300 courses.

## Held constant

- The corpus and text template fixed in part 2. Text is title + newline + description, title alone
  when the description is empty.
- UMAP: n_neighbors 15, min_dist 0.05, cosine, 2 components. Seed 42 is the pipeline's layout;
  seeds 43 and 44 are added for the stability rows. Parameters are not tuned per model.
- Neighbourhoods use cosine distance on the unit-normalised vectors, excluding the course itself.

## Metric

**k-nearest-neighbour label agreement.** For each labelled course, the fraction of its k nearest
labelled neighbours that share a label with it, averaged over labelled courses. Neighbours are
searched within the set of courses that carry the label type in question, so that "no label" is
never counted as disagreement. A course with several tags agrees with a neighbour if they share
any tag.

- Spaces: the embedding space (primary), and the 2-d UMAP layout (secondary), the latter computed
  at three seeds and reported as mean and range.
- k = 15, matching UMAP's n_neighbors (primary); k = 50 as a robustness row.
- Uncertainty: a paired bootstrap over courses (1,000 resamples) of the difference candidate minus
  incumbent, reported as a 95% percentile interval. A shuffled-label null (100 permutations) is
  reported for each label type so the floor is visible.

## Labels

- **Primary: disciplinary breadth category.** Courses whose `gers` list includes any of
  GER:DB-Hum, GER:DB-SocSci, GER:DB-NatSci, GER:DB-Math, GER:DB-EngrAppSci. This is the
  registrar's own subject label, assigned across departments, on the subset of courses that carry
  one. The count of labelled courses is reported.
- **Secondary: department code** (254 values) and **school** (9 values), on every course.
- **Tertiary, informational: WAYS tags** (`WAY-*`), any-overlap rule.

None of the three appears in the embedded text by design. Descriptions occasionally name their own
discipline; that is content, and it is the same for every model.

## Decision rule

1. Compute the primary metric (embedding space, k = 15, breadth category) for the incumbent and
   each candidate.
2. A candidate beats the incumbent if the paired-bootstrap 95% interval of (candidate minus
   incumbent) excludes zero and the point estimate is positive.
3. Among candidates that beat the incumbent, choose the highest point estimate. If two are inside
   each other's paired interval, choose the smaller model.
4. If no candidate beats the incumbent, keep the incumbent.
5. Consistency check: the chosen model's 2-d secondary metric, averaged over three seeds, must not
   fall below the incumbent's by more than the incumbent's own seed range. If it does, keep the
   incumbent and report the conflict.
6. The winner becomes `EMBED_MODEL_KEY`. The downstream pipeline is rerun once. The choice is not
   reopened on the basis of how the resulting map looks; impressions are reported, not decisive.

## Reported, not decisive

- Secondary and tertiary labels; the k = 50 row.
- The same metrics on the raw 11,500-course corpus before the part-2 rules, so a reader can see
  whether the exploratory decisions changed the ranking.
- Neighbourhood overlap between models: Jaccard of 15-nearest-neighbour sets, in the embedding
  space and in the 2-d layouts. This says how much the choice matters, not which is better.
- Trustworthiness (k = 15) of each 2-d layout with respect to its own embedding space: how much the
  layout lies, per model.

## Limitations named in advance

- UMAP parameters are the incumbent's and are not tuned per model; equal treatment, but it could
  disadvantage a model whose local density differs.
- Department purity rewards mimicking the org chart, which the text deliberately omits. That is why
  it is secondary.
- Stability is reported, not ranked on: a space that collapses to a few blobs is very stable.
- Rules developed during the exploratory phase could preferentially remove the incumbent's failure
  modes. The content-only constraint, the second exploration model, and the raw-corpus row bound
  that risk; they do not make it zero.

## Exploratory phase (declared)

Before part 2 is written, maps are built from the incumbent and from `arctic-l-v2`, both named by
Toponymy, and explored to find classes of courses to exclude, combine, or transform. Rules that
come out of this must satisfy three constraints:

1. Expressed in terms of course content or metadata, never map position. The map points at
   candidates; the rule is written against the data and justified there.
2. Applied identically to every model.
3. Logged with counts (the pipeline's `drops.csv` and per-rule report).

Observations that can only be described geometrically become display treatments, such as a flag
colormap, not exclusions. Part 2 lists every rule, the model(s) whose map surfaced it, and its count.

## Procedure

1. Commit part 1 (this document).
2. Exploratory phase on the two maps.
3. Commit part 2: corpus rules and text transforms.
4. Run `pipeline/compare_embeddings.py` once; it writes `docs/embedding_comparison.md`.
5. Apply the decision rule; set `EMBED_MODEL_KEY`; rerun stages 03 to 05.

## Part 2: corpus definition

Written 2026-09-05 after the exploratory phase. No comparison metric had been computed. The rules were
developed while looking at two exploration maps built from the raw 11,500-course catalog, one from the
incumbent (`qwen3-0.6b`) and one from `arctic-l-v2`, each named by Toponymy, with
`experiments/explore_regions.py` describing every finest-layer region by content and metadata. Both
maps surfaced every class below; none of the rules is specific to one model's layout.

The rules are implemented in `pipeline/01b_clean.py`, applied in order so that each course is counted
under one rule, and every removed course is written to `data/stanford/corpus_drops.csv` with its rule.
Thresholds are in `pipeline/config.py`. The rules are written against course content and metadata only.

| rule | definition | threshold | result |
|---|---|---|---|
| 1 placeholder | description shorter than N words | N = 3 | 181 courses dropped |
| 2 administrative | whole description shared verbatim (lower-cased, punctuation and whitespace normalised) by at least M courses across at least D departments | M = 5, D = 5 | 228 courses in 12 groups dropped |
| 3 template | an opening run of whole sentences, at least W words long and shared by at least M courses, is removed from the embedded text; if nothing remains the title stands alone | W = 10, M = 5 | 148 courses stripped, 91 title-only; all kept |

Corpus: 11,500 courses in, **11,091 kept**. The hovercard always shows the full catalog
description; rule 3 changes only the text that is embedded (`embed_text` in `corpus.parquet`).

**How the thresholds were set.** Rule 1: one- and two-word descriptions are "TBD", "(Staff)", "TGR
Dissertation", "TGR Project" and the like; nothing that short describes a course. Rule 2: the verbatim
groups shared by five to nine courses across five or more departments are all of the same kind as the
larger ones ("doctoral practicum in teaching", "TGR project", "CPT course required for international
students"), so the count threshold is five rather than ten; spread across departments is the defining
property of administrative boilerplate. Rule 3: the ten-word floor keeps generic short openings such as
"Prerequisite: consent of instructor." from being stripped when a course-specific description follows.

**What rule 2 removed** (whole description, courses / departments): "Prerequisite: consent of
instructor." 42; "May be repeated for credit." 37; "Students undertake investigations sponsored by
individual faculty members..." 35; "Provides an opportunity for student and faculty interaction..."
(Medical Scholars) 34; "Prerequisite: for grad students only and consent of instructor" 29; and seven
groups of five to seven courses each: investigations with consent, doctoral practicum in teaching,
doctoral practicum in research, repeatable with consent, admitted-to-candidacy dissertation units, CPT
course, observational clinical experience.

**What rule 3 stripped** (opening text, courses): PWR 2 programme paragraph 61; PWR 1 programme
paragraph 48; music lessons for majors 26 (title-only); Frosh 101 24 (title-only); music lessons by
audition 22 (title-only); EE curricular practical training 9; Law School exchange programmes 8
(title-only); language grammar template 6; first-year JD curriculum note 6; and six groups of five or
fewer courses (clinical research units, MS&E internships, music repeat-credit notes, an art-studio
prerequisite, a closed-enrolment note).

**Consequences named in advance.** The 91 title-only texts are far shorter than the
rest and may drift together by register rather than subject; they are few enough to accept and are
disclosed on the methodology page. Unscheduled courses (41% of the catalog) are kept: the map is a
map of what the university teaches, and the "Scheduled in 2026-27" colormap makes the distinction
visible. No courses are combined; cross-listings were already collapsed in stage 01.

**Sensitivity row.** The comparison is also run on the raw 11,500-course catalog with the raw
title + description embeddings (`--raw`), reported alongside, so a reader can see whether these rules
changed the ranking.

## Deviations

_None yet._
