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
- [x] ~~**4. Validation checks**, each a short script in `experiments/`, results into the record:
  kNN department and school agreement by course-number level [4.3.1]; department pairs sharing
  cross-listed courses versus centroid distance, controlling for school [4.3.4, 4.2.2 displacement];
  linear probe on the embeddings versus tf-idf for school and department [4.3.5]; department
  centrality in the existing department-centroid similarity graph [4.3.3].~~
  *Landed 2026-09-06:* four scripts on `experiments/validation_common.py` (the map's inputs aligned to the corpus,
  course-number levels at the Bulletin's own breakpoints, department centroids, bootstrap intervals, a within-department
  fixed-effects contrast), names off, no LLM, seconds each except the probe; the pure functions are unit-tested on
  synthetic data in `tests/test_validation_common.py` and `tests/test_validation_checks.py`; every output is a
  `validation_*` file in `data/stanford/`, copied into the record by stage 07. Run with `OMP_NUM_THREADS=1`.
  *Level* (`level_agreement.py`): 1-99 / 100-199 / 200-299 / 300+ from the number in `primary_code`, the Bulletin's
  own breakpoints (it also says Stanford has no standard numbering, so nothing finer); agreement is the share of a
  course's 15 nearest neighbours (cosine in the embedding space, euclidean in the layout) in the same department or
  school; tested on the UG and GR careers, the professional schools' own schemes tabulated only. Department agreement
  is 0.40 at 1-99 and 0.42 at every level above, school 0.67 then 0.73-0.74; the within-department contrasts against
  1-99 are +0.03 (department) and +0.02 to +0.04 (school) at every higher level with intervals excluding zero (one
  layout contrast, department at 300+, touches it), and there is no further rise above 100-199. Size-matched chance is
  0.01 for department, so the step is not a department-size artefact. Pardos & Nam's lower-division effect replicates
  in content as a small step at the 100 boundary, not a gradient.
  *Cross-listing* (`crosslisting_distance.py`): centroids are the unit-norm mean of a department's *single-listed*
  courses (155 departments with ten or more; 10,492 courses), so no cross-listed course contributes to a centroid it
  is compared with, and a single-listed course's own displacement leaves it out; the naive centroid is a sensitivity
  row with the same signs. Of 11,935 department pairs, 851 share a cross-listed course; their centroid distance is
  0.197 against 0.308, a random linked pair is closer than a random unlinked one with probability 0.80 (0.73 among
  same-school pairs, 0.75 among cross-school pairs), the linked coefficient with school-pair fixed effects is -0.089
  [-0.095, -0.083] with a within-school-pair permutation p of 0.001 (the floor at 1,000 permutations), and distance
  falls with the number of shared courses (0.23 / 0.18 / 0.16 for 1 / 2-4 / 5+; Spearman -0.40 among linked pairs).
  Displacement from the home centroid rises with listings, 0.158 / 0.177 / 0.193 / 0.206 for 1 / 2 / 3 / 4+,
  within-department +0.019 / +0.032 / +0.045, and is larger when the listings span schools (0.223 against 0.175).
  The home department, the registrar's `academicOrganization`, is the content-nearest of a cross-listed course's
  listed departments 43% of the time against 44% for a random pick among them: cross-listed courses sit between their
  departments, not in the owner's core. `crosslisting_displacement.parquet` is per course, aligned to the corpus, for
  step 6.
  *Probe* (`linear_probe.py`, nine minutes, five outer folds in parallel): LinearSVC (one-vs-rest, squared hinge) on
  the unit-norm embeddings and on tf-idf over the same embedded text (unigrams and bigrams, min_df 2, sublinear tf, fit
  per training fold), C from {0.1, 1, 10} by 3-fold inner cross-validation, 5-fold stratified outer, on the 10,729
  courses in the 165 departments with ten or more. School (9 classes, majority 0.50): embeddings 0.827, tf-idf 0.837,
  macro-F1 0.78 both. Department (165 classes, majority 0.06): 0.659 against 0.656, macro-F1 0.60 both, fold sd under
  0.01; about half of the wrong department predictions land in the right school (0.51 / 0.52). Open Syllabus's
  bag-of-words result replicates: a linear bag-of-words classifier recovers the org chart from descriptions as well as
  the embedding does, and both leave a third of departments unrecovered from content (`validation_probe_departments.csv`
  has per-department recall; interdisciplinary programmes and language departments are the misses, professional and
  performance departments the hits). The inner search picked C = 10, the top of the grid, in 18 of 20 folds, with the
  gain from 1 to 10 under 0.005, so a wider grid would not move the numbers. Cross-validated predictions per course are in
  `validation_probe_predictions.parquet`.
  *Centrality* (`department_centrality.py`): 140 departments with 15 or more courses, centroid cosine similarity;
  strength (mean similarity to the others), eigenvector centrality, in-degree and closeness in the 5-nearest-department
  graph, the last two again in the 2-d layout, and a 200-draw course bootstrap for the strength rank. The core by
  strength is the cross-cutting units: the overseas-studies programmes, the Master of Liberal Arts, the Division of
  Literatures, Cultures & Languages, Stanford in New York, Symbolic Systems, CSRE; strength tracks within-department
  spread (Spearman 0.63), a heterogeneous unit's centroid sitting near the global mean. By school, VPUE leads
  (probability of higher strength than a department outside the group 0.91) and Medicine and Athletics trail (0.22,
  0.03); Engineering is mid-table by strength (mean rank 85 of 140, probability 0.39) but ties VPUE for the highest
  in-degree (6.2). Within H&S, mapped to its three divisions by the department list on Wikipedia's page for the school
  (the school's own site names the divisions but did not list members on the pages fetched; mapping recorded in the
  JSON, a small hand mapping accepted where the 254-to-20 one of 4.3.2 was cut), the Natural Sciences are the
  periphery: mean strength rank 117 of 140, probability 0.15, p 0.002; the Social Sciences are central (0.73). Gim et
  al. half-replicates: natural sciences peripheral, yes; engineering core, only by in-degree. The measures disagree
  (in-degree against strength, Spearman 0.44), so the CSV keeps all of them.
- [ ] **5. Archive with a DOI** [4.4.2]. Gate: read the ExploreCourses terms of use first. If
  redistribution is allowed, Zenodo deposit of the raw XML snapshot, embeddings, corpus parquet, and
  the record; if not, the deposit drops the raw XML and the write-up says why. DOI into the README
  and the record.
  *In progress, 2026-09-06.* Gate read: ExploreCourses' footer links Stanford's site terms of use
  (stanford.edu/site/terms), which allow downloads "only for User's own personal, non-commercial use" and forbid
  otherwise copying, reproducing, distributing or publishing material; its About page says the course data API
  "is available to students and faculty". ExploreCourses itself returned 503 at every attempt that day, so its
  pages were read from the Internet Archive capture of 2026-08-29 and the Stanford terms page live. Outcome: the
  raw XML is not deposited. `pipeline/08_archive.py --checksums` writes `raw_checksums.json` (256 files, 227 MB,
  SHA-256 each, the fetch manifest, the terms finding) so that a copy obtained from Stanford can be verified
  against the snapshot the map was built from, and stage 07 puts it in the record. The corpus and catalog
  parquet files stay in the deposit as the plan lists them: they carry the same descriptions the public map and
  the committed record already do, a judgment the write-up states rather than hides. The stage (`--checksums`,
  `--reserve`, `--upload`, `--publish`; `--sandbox` for a rehearsal, `--dry-run` to print the requests; pure
  parts unit-tested in `tests/test_archive.py`) builds a 19-file, 551 MB deposit: the six primary artifacts
  (corpus, catalog, labels, layout, names, tree), the eight candidate-model embeddings, the record directory as
  one tarball, an exploration tarball (the raw-catalog maps, the other candidates' layouts, the EVoC layers),
  the repo at the archived commit, a README with every file's SHA-256, and `raw_checksums.json`. Metadata:
  dataset, CC BY 4.0, version 1.1, related identifiers for the repo, the live map and ExploreCourses, DOI
  pre-reserved so it can go into the README and the record before publishing. `archive.json` (host,
  deposition, DOI, per-file checksums, the terms finding) is in the record, and so, from now on, is the
  published layout `umap_coords.npz`, which the record had left to the map HTML. The map HTML is not in the
  deposit: step 6 rebuilds it and changes nothing the record rests on. *Remaining:* a Zenodo token
  (`ZENODO_TOKEN` exported in `~/.secrets`; `ZENODO_SANDBOX_TOKEN` to rehearse), then `--reserve`, the DOI
  into README.md, stage 07, commit, `--upload`, `--publish` (irreversible), stage 07, commit. *Reserved 2026-09-06:*
  deposition 22550382 on zenodo.org, DOI 10.5281/zenodo.22550382, in the README and the record; upload and publish pending.
  *Open before upload (2026-09-06 discussion):* whether the deposited parquet files keep the full descriptions
  (the map already ships them; the Course-Skill Atlas precedent released derived features only) or carry a
  per-course description hash instead; whether to ask the Registrar's webmaster for permission in parallel;
  and dropping the `instructors` column from the deposit, which plays no part in the analysis.
  *Decided 2026-09-06 (the IP discussion; see the decisions below):* the deposit is derived data only. The
  catalog and corpus parquet files lose their prose and people columns (`description`, `embedded_description`,
  `embed_text`, `learning_objectives`, `instructors`) and gain a per-course `description_sha256`, so a copy
  fetched from Stanford can be matched row by row; titles, every metadata column and the rule columns
  (`embed_text_rule`, `stripped_words`, `template_shared_by`) stay. Embeddings, layout, labels, names, tree,
  record and code are unchanged; the license statement is CC BY 4.0 on the derived data, with the titles named
  as Stanford's. The full-text corpus goes into a second Zenodo deposition with `access_right: restricted` and
  an `access_conditions` statement (personal, non-commercial research use, the use Stanford's terms allow),
  each record naming the other as a related identifier, so a reproducer can request the text once
  ExploreCourses is gone (it returned 503 again on 2026-09-06). Zenodo's access right is per deposition, which
  is why it is two records rather than one with mixed files. The committed record gets the same treatment: the
  tracked `docs/record/stanford/corpus.parquet` is stripped of the same columns and stage 07 copies it stripped
  from now on; the old blob stays in git history, which the write-up says rather than rewriting public history.
  `corpus_drops.csv` keeps its description column: its 409 rows are the placeholders and the twelve boilerplate
  texts of rules 1 and 2, and the file is the audit of those rules. Checked 2026-09-06: no other record file
  carries description text (the listener call logs store id, answer and model only). The map is unchanged,
  hover excerpt and search field included, a judgment the write-up states. The Registrar is not asked for
  v1.1: a derived-only deposit needs no permission, and asking is worth it only for a v2 corpus meant for
  redistribution, for every university at once. *Remaining:* the stage-07 and stage-08 changes above, the
  stripped record committed, `--upload`, `--publish`, the restricted sibling record, stage 07, commit.
- [ ] **6. Stage-05 rebuild, once.** Cross-listing count colormap [4.2.2]; course-number level
  colormap [4.2.8]; footer link to the repo and to the reserved blog-post slug; region stability
  [4.2.3] in whichever of hover text and continuous colormap survives a look at the rebuilt map.
  Reserve the slug before this step so the map is rebuilt only once.
  *Decided 2026-09-06:* the rebuild also moves the map to `docs/stanford/` (a redirect stays at the root), so
  that a later index over several catalogs breaks none of the links the post and the Zenodo record carry; the
  v1.1 build keeps that path when v2 adds a shared-space map, and the record's live-map identifier is updated
  by metadata edit, as for the tag.
- [ ] **7. Tidy.** Verify the click-through in a browser for both the Navigator link and the
  ExploreCourses fallback; fix the stale "4B comparison on the table" line in `CLAUDE.md`; verify or
  cut the three lit-review characterisations that rest on recollection (Börner et al. 2018's third
  corpus, Biasi & Ma's gap measure, Wang et al. 2021's venue); tag the release; file the Toponymy
  per-event-loop semaphore bug upstream.
  *Partly landed, 2026-09-06.* Click-through: the Navigator link verified in a browser (ORALCOMM 117 opens
  navigator.stanford.edu/classes/1274/24952, its Winter 2027 class page); the ExploreCourses fallback could not
  be verified because explorecourses.stanford.edu returned 503 all day (the Internet Archive's last 200 capture
  is 2026-08-29). Re-check when it is back; if the outage persists, step 6 should weigh a different fallback for
  the 4,912 unscheduled courses. The stale 4B line is out of `CLAUDE.md`. The three lit-review
  characterisations were checked against the sources and corrected in place: Börner et al. 2018's corpora are
  publications, course syllabi and job advertisements (the review said "course offerings"); Biasi & Ma's gap is
  100 times the ratio of a syllabus's mean cosine similarity to older articles (τ years before it, a three-year
  window) to that with recent ones (τ' years), τ and τ' the field's 90th and 5th percentile citation lags, and
  NBER 29853 was revised in August 2026 as "Frontier Knowledge in Higher Education" with the measure renamed
  frontier knowledge proximity; Wang et al. 2021 is JMLR 22(201):1-73. The Toponymy issue is drafted (the
  `AsyncLiteLLMNamer` semaphore binds to the first `asyncio.run` loop; a reproduction without an API key; the
  per-loop fix) and waits on a go to file. *Remaining:* file the issue; verify the fallback; tag. *Decision:*
  tag after step 6, so the tag marks the completed map; the Zenodo record names the archived commit and takes
  the tag as a related identifier through a metadata edit, which needs no new version.
- [ ] **8. Blog post, then promote.** The post is the methodology page. It must carry: region names
  are LLM-generated; the claimed-versus-practised and descriptions-versus-syllabi caveats [4.1.4]; the
  corpus rules with counts; the preregistration outcome including the 4B tie; the Chari-Pachter debate
  answered with the step-1 numbers; the note that collapsed cross-listings make the Pardos validation
  set unusable as written [4.1.6]; the stability and intrusion results; the DOI. Then share with the
  reader groups in [4.4.4].
  *Decided 2026-09-06:* not held for the multi-university v2. This is the methods-and-record post for the
  first catalog; it says more may follow and promises no design. A v2 post would be a findings post that
  cites it.

## Deferred, not part of completion

- Interdisciplinarity index per course [4.2.1]: held back, not cut. A new derived quantity that would
  need its own methods paragraph and validation.
- Prerequisite structure [4.2.4]: about 1,860 of 11,091 descriptions carry a "Prerequisite:" clause,
  so a colormap would be mostly "none". Cut.
- Han discipline similarity matrix [4.3.2]: needs a hand mapping of 254 departments to 20 disciplines.
  Cut for effort.
- Anything needing new data or a new inference layer: O*NET work-activity layer [4.2.5], SDG tags
  [4.2.6], research-frontier proximity [4.2.7], first year offered [4.2.9], longitudinal Stanford
  [4.4.1, 4.3.6], second institution [4.4.3], which is now the v2 decision below. Follow-on projects.

## Decisions taken while planning (2026-09-05)

- Blog post is in scope and comes last; it cites the DOI, so the archive precedes it.
- One human rater for the intrusion task is accepted as a compromise and disclosed as such.
- Zenodo DOI rather than a release asset alone, gated on the terms of use.
- Region confidence: compute first, then choose among blog figure, hover text, and colormap on the
  rebuilt map. All three are cheap once the step-1 parquet exists.
- (2026-09-06) "Wayfinding lineup" means the discussion-177 instrument, an identification task on course text;
  the step-1 seed layouts are not its nulls and stay available for step 6's display options.
- (2026-09-06) The deposit excludes the raw XML (Stanford's terms) and the map HTML (step 6 rebuilds it) and
  keeps the corpus and catalog parquet files; the release tag follows step 6 rather than step 7's order.
- (2026-09-06) **Stanford stays for v1.1.** The IP concern step 5 raised is about the deposit, not the source.
  The alternatives considered (Cornell, UIUC, MIT) expose or publish an API without any license to
  redistribute the text, so the derived-only posture would be the same there; only Waterloo has an explicit
  open-data license. Silence is not permission. The larger Stanford risk is the source's survival
  (ExploreCourses is the heritage catalog and was down again on 2026-09-06), which bears on v2, not on a
  snapshot with a DOI.
- (2026-09-06) **Multi-university is v2**, planned in its own doc after step 8 and kept out of this plan (a
  guard against expanding scope until the first version is never shared). Three catalogs, not five. A
  selection rule written before fetching: a documented public endpoint required, an explicit license
  preferred, Stanford grandfathered as the disclosed exception with its text withheld. Steven's lean is the
  shared-embedding-space comparison plus a map per university (the gallery is cheap once the comparison
  exists and shows what the shared view cannot), with the gallery alone as the fallback if the comparison
  proves less feasible or less interesting than expected. Known hazards for the design doc: registrar house
  style as a batch effect, and a common corpus definition (Cornell's roster is term-scoped).
- (2026-09-06) The blog post is not held for v2 (step 8), and the v1.1 map keeps a stable path (step 6).
