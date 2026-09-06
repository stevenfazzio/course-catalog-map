# Completion plan (v1.1)

Written 2026-09-05. What remains before the project is called complete. References in brackets point
to section 4 of `docs/lit_review.md`. Strike items through as they land; record deviations in place.

## Definition of complete

The map, an auditable record, and one public write-up, promoted. Not a research paper. Two cut rules:

1. **Layout and names are frozen.** Naming cost money and is not bit-reproducible across machines.
   Anything that touches only stage 05 (colormaps, hover, footer) is in scope; anything upstream is v2.
2. **An analysis is in scope only if the write-up makes a claim that needs it.** "The regions are
   real" needs stability and the Kobak-Berens triple. "Content recovers the org chart to degree X" is
   already answered by the preregistered comparison. Everything else is a follow-on post or project.

## Steps, in order

- [x] ~~**1. Record metrics and stability.** One experiments script, names off, output into
  `docs/record/stanford/`. Kobak-Berens triple: KNN preservation, KNC over departments and schools,
  CPD [4.1.1]. One UMAP run at a larger `n_neighbors`, KNC compared, reported not adopted [4.1.5].
  Stability [4.1.2]: ten UMAP seeds on the full corpus and ten 80% subsamples at the published seed,
  Toponymy clustering only. Per region and layer, mean pairwise co-assignment of published members;
  per course, the fraction of its published-region peers it stays with; per layer, AMI against the
  published clustering. One parquet aligned to the corpus so that every display option in step 6 is
  possible from it.~~
  *Landed 2026-09-05:* `experiments/record_metrics.py` (metric functions unit-tested on synthetic data in
  `tests/test_record_metrics.py`), 3.7 minutes locally, no LLM. Outputs, copied into the record by stage 07:
  `layout_metrics.json`, `stability.parquet` (per course), `stability_regions.csv` (per region),
  `stability_meta.json` (run record), `stability_layouts.npz` (every rerun's coordinates and cluster
  layers, so step 2 has its null layouts and step 6 needs no new runs). Choices made in implementation:
  KNN at k = 10 (Kobak-Berens) and 15 (the pipeline's); KNC at k = 10 over all 237 departments present
  and again over the 121 with 20+ courses, k = 3 over the nine schools; CPD on five draws of 1,000
  courses; embedding-space distances are cosine throughout, as UMAP saw them. Published layers are
  matched to rerun layers by nearest region count (one seed produced a fifth, six-region layer, left
  unmatched). A pair the rerun leaves unclustered is not co-assigned; AMI is on courses both partitions
  cluster, with the share of published members the rerun unclusters reported next to it. Two additions
  beyond the plan: a zero-perturbation control (the published layout re-clustered on this machine, since
  the clusterer is deterministic per machine but not across machines) and the triple for every seed
  layout. Headline: KNN 0.45, KNC 0.45 / 0.54 / 0.67 (departments / large departments / schools), CPD
  0.49, trustworthiness 0.99; `n_neighbors` 100 drops KNN to 0.34 for CPD 0.53 with KNC flat, so 15
  stands. Seeds: AMI 0.92 / 0.94 / 0.91 / 0.86 by layer, finest first; subsamples 0.89 / 0.91 / 0.88 /
  0.82; control 0.99+. Median region co-assignment 0.68 at the finest layer, 71 of 357 regions above
  0.9 and 10 below 0.2; the coarsest layer splits its large regions (median 0.60).
- [x] ~~**2. Wayfinding lineup** (Wickham et al.), sampled, reusing the step-1 seed runs as the null
  layouts.~~
  *Landed 2026-09-06:* `experiments/wayfinding_lineup.py`, the instrument from Steven's Toponymy label-quality
  study (TutteInstitute/toponymy discussion 177): a listener model sees a region's name and five candidate
  regions, the region and its four nearest same-layer neighbours by centroid cosine in the embedding space, each
  shown as five held-out member courses (title and description excerpt, no codes); the name's score is the
  probability mass the listener puts on the true region. *Deviation:* the step-1 seed layouts are not used. The
  task shows course text, not positions, so a null layout has no role in it, and a graphical lineup of the
  published layout among seed layouts would test the inverse of the usual hypothesis (a distinguishable panel
  would mean instability), which step 1's co-assignment already measures. Choices: listener Claude Sonnet 5
  (the namer was Opus 5, so no same-model listener; same family, disclosed), effort medium, structured output,
  eight concurrent calls with a resume cache; 136 items (forty regions at layers 0 and 1, every region at 2
  and 3); gold three times with candidate order reshuffled (position bias marginalised, repeat band measured),
  a shuffled-name floor (a random non-candidate same-layer name) and a distant-distractor ceiling; held-out
  documents replay Toponymy's central exemplar selection (eight per region) and exclude those courses, which
  eleven true regions were too small to allow in full (they scored lower, 0.51 against 0.58, so no inflation).
  Result: probability mass 0.58 [0.55, 0.60] against chance 0.20; per layer 0.53 / 0.59 / 0.58 / 0.64 finest
  to coarsest; top-1 0.93; no item below chance; shuffled 0.18, distant 0.93; repeat band p90 0.12; mean
  score by position flat. Lowest names: "Warfare Society And Asia-Pacific Historical Transformation" (0.20),
  "Faculty-Supervised Doctoral Dissertation Research" (0.23), "European Languages and Linguistics" (0.25).
  One item dropped: the listener's safety classifier refused every call on "Emerging Viral Diseases And
  Biosecurity" (category bio), recorded in the JSON. 680 calls in two runs, about $5.50. The numbers sit where
  the same instrument put gold labels on 20 Newsgroups and arXiv (0.55 to 0.57 gold, 0.12 to 0.18 shuffled,
  0.81 to 0.94 distant).
- [x] ~~**3. Name intrusion task** [4.1.3]: fifty items, four names from one region layer plus one
  intruder, layers mixed and order shuffled. Steven is the single rater and has seen the map; both
  facts are disclosed in the record next to the result.~~
  *Landed 2026-09-06:* Steven rated all fifty: 45 correct (0.90, Wilson 95% [0.79, 0.96], exact binomial p
  against chance 0.20 of 2.5e-26); by child layer 35/37, 7/10, 3/3 finest first; parent-unit items 36/40,
  grandparent-unit 9/10. The listener (Sonnet 5) scored 48/50 on the same items and made the same pick as
  Steven on 45; both missed I15 and I18, the listener alone got three more, Steven alone none. Three of
  Steven's five misses had an intruder whose own parent sits next to the item's ancestor in content (criminal
  justice among global-history names, oceanography among climate-policy names, Spanish among Asian-language
  names): the intruder rule is "different coarsest ancestor", a tree criterion, and does not guarantee
  semantic distance, so those items read as hard rather than as bad names. Record: `name_intrusion.json`,
  with the answers, key, items and rater page beside it. *Instrument, built earlier the same day:* `experiments/name_intrusion.py`. The
  plan left the coherence unit of the four names open; the parent region is the direct analogue of Chang et
  al.'s topic, so an item is four children of one region at one layer plus one same-layer region under a
  different coarsest ancestor. Parents with four or more same-layer children give 40 items (several parents'
  children are split across layers); the other ten are four descendants of one grandparent drawn through at
  least two of its children, chosen at random. By child layer 37 / 10 / 3; the 14 coarsest regions have no
  sibling structure above them and are tested by the lineup instead. Item order and intruder position are
  shuffled (seed 0). To rate: serve `data/stanford/` on port 8765, open `name_intrusion_rater.html`, answer all
  fifty, copy the answers into `data/stanford/name_intrusion_answers.csv`, run `--score`, then stage 07; do not
  open `name_intrusion_key.json` first. The listener (Sonnet 5) has already answered the same items, 48 of 50,
  missing "Shipboard Oceanography and Estuarine Biogeochemistry" among climate-policy names and "AI Regulation
  Privacy Law Practicums" among law-and-politics names; `--score` adds human-listener agreement to the record,
  which is the grounding for using the listener in step 2 [Krumdick 2025].
- [ ] **4. Validation checks**, each a short script in `experiments/`, results into the record:
  kNN department and school agreement by course-number level [4.3.1]; department pairs sharing
  cross-listed courses versus centroid distance, controlling for school [4.3.4, 4.2.2 displacement];
  linear probe on the embeddings versus tf-idf for school and department [4.3.5]; department
  centrality in the existing department-centroid similarity graph [4.3.3].
- [ ] **5. Archive with a DOI** [4.4.2]. Gate: read the ExploreCourses terms of use first. If
  redistribution is allowed, Zenodo deposit of the raw XML snapshot, embeddings, corpus parquet, and
  the record; if not, the deposit drops the raw XML and the write-up says why. DOI into the README
  and the record.
- [ ] **6. Stage-05 rebuild, once.** Cross-listing count colormap [4.2.2]; course-number level
  colormap [4.2.8]; footer link to the repo and to the reserved blog-post slug; region stability
  [4.2.3] in whichever of hover text and continuous colormap survives a look at the rebuilt map.
  Reserve the slug before this step so the map is rebuilt only once.
- [ ] **7. Tidy.** Verify the click-through in a browser for both the Navigator link and the
  ExploreCourses fallback; fix the stale "4B comparison on the table" line in `CLAUDE.md`; verify or
  cut the three lit-review characterisations that rest on recollection (Börner et al. 2018's third
  corpus, Biasi & Ma's gap measure, Wang et al. 2021's venue); tag the release; file the Toponymy
  per-event-loop semaphore bug upstream.
- [ ] **8. Blog post, then promote.** The post is the methodology page. It must carry: region names
  are LLM-generated; the claimed-versus-practised and descriptions-versus-syllabi caveats [4.1.4]; the
  corpus rules with counts; the preregistration outcome including the 4B tie; the Chari-Pachter debate
  answered with the step-1 numbers; the note that collapsed cross-listings make the Pardos validation
  set unusable as written [4.1.6]; the stability and intrusion results; the DOI. Then share with the
  reader groups in [4.4.4].

## Deferred, not part of completion

- Interdisciplinarity index per course [4.2.1]: held back, not cut. A new derived quantity that would
  need its own methods paragraph and validation.
- Prerequisite structure [4.2.4]: about 1,860 of 11,091 descriptions carry a "Prerequisite:" clause,
  so a colormap would be mostly "none". Cut.
- Han discipline similarity matrix [4.3.2]: needs a hand mapping of 254 departments to 20 disciplines.
  Cut for effort.
- Anything needing new data or a new inference layer: O*NET work-activity layer [4.2.5], SDG tags
  [4.2.6], research-frontier proximity [4.2.7], first year offered [4.2.9], longitudinal Stanford
  [4.4.1, 4.3.6], second institution [4.4.3]. Follow-on projects.

## Decisions taken while planning (2026-09-05)

- Blog post is in scope and comes last; it cites the DOI, so the archive precedes it.
- One human rater for the intrusion task is accepted as a compromise and disclosed as such.
- Zenodo DOI rather than a release asset alone, gated on the terms of use.
- Region confidence: compute first, then choose among blog figure, hover text, and colormap on the
  rebuilt map. All three are cheap once the step-1 parquet exists.
- (2026-09-06) "Wayfinding lineup" means the discussion-177 instrument, an identification task on course text;
  the step-1 seed layouts are not its nulls and stay available for step 6's display options.
