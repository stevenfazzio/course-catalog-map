# Literature review: course catalogs as a corpus, and what it implies for the map

Working review, entries verified progressively. Three characterisations rested on recollection at compile
time (the third corpus in Börner et al. 2018, the exact form of Biasi & Ma's gap measure, the venue of Wang
et al. 2021); all three were checked against the sources on 2026-09-06 and corrected in place.

Compiled 2026-09-05 from OpenAlex, Semantic Scholar, and the open web. Scope: work that uses course
catalogs, course descriptions, or syllabi as a dataset; the map-of-science and embedding-visualisation
literature the pipeline sits in; and the evaluation literature for 2-d layouts, clusterings, and
LLM-written labels. Items marked (abstract) were read at abstract level only; the rest were read in
full or at methods level. Reference keys in brackets point to the list at the end.

The one-paragraph summary: nobody has published a content-embedded map of a whole university catalog.
The closest precedents are an enrollment-embedded t-SNE map of Berkeley courses [Pardos & Nam 2020], the
Open Syllabus Galaxy (node2vec + UMAP over 1.1M syllabi's reading lists) [McClure 2021], and a
network-based knowledge map of Korean STEM departments built on LLM-standardised course names [Gim et
al. 2025]. Course descriptions are used heavily as a corpus, but almost always instrumentally: to
measure interdisciplinarity [Han et al. 2023; Hong et al. 2025], to align teaching with labour-market
skills [Chau et al. 2023; Javadian Sabet et al. 2024; Light 2026], to predict prerequisites and
equivalencies [Liang et al. 2017; Jiang & Pardos 2020; Kim et al. 2024], or to classify courses into
external schemes such as SDGs [Kharlashkin et al. 2024; Prior et al. 2024]. That leaves the map itself
as the contribution, and the surrounding literature as a source of (a) validation targets the map can
reproduce, (b) per-course quantities that would make good colormaps, and (c) evaluation conventions
the record should adopt.

## 1. Closest precedents

**A university map of course knowledge** [Pardos & Nam 2020]. Skip-gram ("course2vec") embeddings of
UC Berkeley courses trained on 2.1M enrollment sequences from 124k students (2008 to 2016), 7,997
lecture courses filtered to 4,349 with 20+ enrollments, laid out with Barnes-Hut t-SNE. Validation is
the interesting part: 1,472 cross-listed pairs and 381 credit-equivalency pairs scored by nearest-
neighbour rank, 2,256 course analogies (40% accuracy overall), and multinomial regressions predicting
six registrar attributes from the vectors (subject 84%, all attributes 88% versus a 31% majority-class
baseline). They also regress the enrollment vectors onto bag-of-words catalog descriptions to describe
regions with keywords. A finding worth checking on Stanford: lower-division courses formed an
unstructured cloud near the origin while upper-division courses showed disciplinary structure.
Enrollment data is FERPA-restricted; the embeddings reflect who takes what, not what is taught, which
is exactly the axis the Stanford map is on the other side of.

**Sources of course information** [Jiang & Pardos 2020]. Same group, four representations (syllabus
bag-of-words, catalog-description bag-of-words, course2vec, enrollment graph embeddings) on four tasks.
Catalog descriptions were the best single representation for course similarity and credit
equivalency; syllabi beat descriptions for prerequisite prediction; enrollment graph embeddings beat
both there. Supports the design choice of description text for a similarity map, and names its limit.

**Open Syllabus Galaxy v2** [McClure 2021; Open Syllabus docs]. node2vec (64-d) on the syllabus-to-work
bipartite graph (1.14M syllabi, 2.28M works), UMAP with n_neighbors=100 and 1,000 epochs, WebGL (regl)
rendering with a 500k-point foreground tile and Elasticsearch-backed background loading. Works are
coloured by the field in which more than 50% of their assignments occur; multi-field works are grey
and labelled "Multiple Fields". No region names, no formal layout evaluation. The Open Syllabus
pipeline itself is documented: 10M deduplicated syllabi, a LinearSVC field classifier over 70 CIP-derived
fields at 84% test accuracy, and a DistilBERT extractor for description and learning-outcome fields.
Their note that bag-of-words does "fairly well" for field classification is a useful baseline claim.

**Knowledge maps of Korean STEM curricula** [Gim et al. 2025]. 126,437 course entries from 2,841 STEM
departments at 161 institutions (spring 2024), course names only. Gemini 1.5 Pro tokenises names and
builds hyperedges that collapse them to 24,476 standardised courses; departments are then compared by
weighted Jaccard over shared courses, communities found with Louvain at four resolutions, and a
"curriculum synergy score" (Jensen-Shannon divergence between two fields' curricula times their
Jaccard similarity) ranks field pairs. Finding: engineering departments form the core of the
similarity network and basic sciences sit on the periphery. LLM used for normalisation, not naming.

**Interdisciplinary college curriculum** [Han et al. 2023]. Catalog descriptions from 80 US colleges
(69,099 courses, 2018) and 110,421 Open Syllabus syllabi for the same schools; 20 disciplines. Each
discipline is a tf-idf document; pairwise discipline cosine similarities define distance; a course's
interdisciplinarity index rewards similarity to several mutually distant disciplines. Sociology-
anthropology similarity 0.39, English-physics 0.01. Catalogs ("claimed" curriculum) show elite liberal
arts colleges claiming far more interdisciplinarity than syllabi ("practised") bear out, and only
practised interdisciplinarity predicts earnings (about +4% per index unit for science majors).
Replication data on OSF. This is the paper to cite for the catalog-versus-syllabus caveat, and its
course-level index transfers directly to embedding space.

**Tooling in the same genre.** Embedding Atlas [Ren et al. 2025] and Nomic Atlas are general-purpose
embedding-map viewers with automatic clustering and labelling; the UCSD map of science [Börner et al.
2012] and VOSviewer [van Eck & Waltman 2010] are the science-mapping ancestors, and Klavans & Boyack
[2017] is the standard reference for asking which of several taxonomies of a corpus is most accurate.
Suominen & Toivanen [2015] compare an unsupervised topic-model map of science to a human-assigned
subject classification, which is the shape of the preregistered breadth-category comparison here.

## 2. Course descriptions and syllabi as a corpus, by purpose

### 2.1 Structure of knowledge and interdisciplinarity
- [Hong et al. 2025] 478,233 syllabi, 2004 to 2019, lexical, topical, and pedagogical analyses; disciplinary
  boundaries were remarkably stable despite institutional rhetoric about interdisciplinarity.
- [Han et al. 2023] as above; [Herzog et al. 2022] apply bibliometric coupling to syllabus reading
  lists to measure curricular interdisciplinarity (abstract); [Yu et al. 2024] propose course-level
  interdisciplinarity indicators for an LIS programme (abstract).
- Sociology of the curriculum, all catalog-based: Frank & Gabler [2006] track the rise and fall of
  fields in university catalogs worldwide across the 20th century; Brint et al. [2008] count
  interdisciplinary degree-granting fields in US catalogs 1975 to 2000; Jacobs [2014] argues from
  cross-listing and joint-appointment data that disciplines remain the organising unit; Abbott [2001]
  is the theoretical frame for fractal self-similar disciplinary structure, which a multi-layer region
  hierarchy is well placed to illustrate.
- Cross-listing is the traditional observable for interdisciplinarity in this literature. The map
  already carries every code per course; it is not yet displayed.

### 2.2 Course representations for prediction, recommendation, and articulation (the Pardos line)
- [Pardos et al. 2019] use enrollment patterns plus catalog descriptions to propose course-to-course
  articulations between institutions; [Kim et al. 2024] do pairwise equivalency with unmodified LLMs over
  public descriptions; [Kwak, Schmucker & Pardos 2026] graph-theoretic articulation networks (abstract).
- [Dong, Yu & Pardos 2019] regress enrollment embeddings onto description bag-of-words to infer latent
  search keywords; [Xu & Pardos 2024] show course *codes* (e.g. PHYS_H101) carry strong similarity
  signal via subword embeddings, a reminder that code-derived features are informative and, by the
  same token, that the org chart leaks through codes if they are ever embedded.
- [Lang et al. 2022] forecast terminal major from course-sequence embeddings at a private university
  (26,892 undergraduates, 2000 to 2020); [Chen et al. 2022] and [Chen, Iyer & Kizilcec 2024] build
  catalog-embedded tools at Cornell; [Chaturapruek et al. 2021] and [Dalberg, Cortes & Stevens 2024]
  are the Stanford Carta/Pathways Lab papers on course consideration and major choice, built on
  Stanford's internal enrollment data. Carta is the natural institutional reader of this map.
- [Liang et al. 2017] recover concept prerequisite relations from course dependency graphs, with a
  dataset of CS courses from 11 US universities; [Xiao, Bai & Wang 2022] extract prerequisite
  relations among concepts from course descriptions directly (abstract).

### 2.3 Skills, tasks, and the labour market
- [Börner et al. 2018] compare skill vocabularies across research publications, course syllabi, and job
  advertisements (millions of each, 2010-2016) and find a soft-skills gap.
- [Chau et al. 2023] and [Javadian Sabet et al. 2024] (Course-Skill Atlas, Scientific Data): 3M+ syllabi
  from ~3,000 institutions; sentences classified as learning content (86% of sentences were logistics
  and dropped), embedded with SBERT all-mpnet-base-v2, and scored against each O*NET detailed work
  activity (DWA) by the *maximum* sentence cosine; aggregated to 281,153 institution-year-field records.
  Validated by field clustering and KL divergence against labour demand. Data cutoff 2017.
- [Light 2026a] "How exposed is higher education to AI": course descriptions from a 1,019-institution
  catalog panel (50M sections since 1996, [Light 2026b]); spaCy verb-object pairs filtered to the
  14,082 pairs that also occur in O*NET DWA text; each pair scored with the Eloundou et al. [2023] LLM
  exposure ratings; field exposure is the frequency-weighted mean. Findings: exposure is broad and
  writing-led rather than concentrated in quantitative fields; average exposure barely moved in 15 years;
  1.1M syllabi from 27 institutions show AI policies in most syllabi by fall 2025 but little change in
  assessment structure; CS enrollment share fell in 2025-26 for the first time in two decades.
- [Yu et al. 2021] framework for education-occupation alignment with NLP; [Xu et al. 2025] benchmark
  LLM skill extraction from 400 curriculum documents: retrieval-augmented prompting wins, zero-shot
  prompting loses to traditional NLP in most settings; [Xu et al. 2026] extend to 21st-century
  competencies; [Musazade, Mezei & Zhang 2026] release UniSkill, an ESCO-to-course matching dataset.
- [Biasi & Ma 2022] "education-innovation gap": for each of 1.7M syllabi, 100 times the ratio of its mean
  cosine similarity to older articles (published τ years before the syllabus, a three-year window) to its
  mean similarity to recent ones (τ' years before), with τ and τ' the 90th and 5th percentiles of citation
  lags in the syllabus's field, over 20M articles; higher means older content. The NBER paper was revised in
  August 2026 as "Frontier Knowledge in Higher Education" and now calls the measure frontier knowledge
  proximity.

### 2.4 Classifying courses into external schemes
- SDGs: [Kharlashkin et al. 2024] distil PaLM 2 labels into small classifiers for course descriptions;
  [Prior et al. 2024] semantic matching of SDGs to curricula (abstract).
- Fields: Open Syllabus (CIP-based, 84%); [Aasheim et al. 2015] contrast data-science and data-analytics
  programmes through their course descriptions; the "count courses about X across catalogs" genre
  (e.g. evolutionary medicine [Grunspan et al. 2019], modern physics syllabi [Buzzell et al. 2025]).

### 2.5 Prerequisite networks
- [Aldrich 2015], [Stavrinides & Zuev 2022] (Caltech), [Yang et al. 2024] (five Midwestern public
  universities): course-prerequisite networks as DAGs, with centralities and chain lengths as curricular
  diagnostics; [Heileman et al. 2018] "curricular analytics" formalises curriculum complexity from the
  same graphs. ExploreCourses has no structured prerequisite field, but descriptions carry
  "Prerequisite:" text.

### 2.6 Longitudinal catalogs
- UC ClioMetric History Project [Bleemer 2018; uccliometric.org]: formatted-OCR digitisation of
  catalogs and directories, including UC *and Stanford* faculty and courses 1900 to 2011, 800k+ course
  name/description records, largely public.
- [Light 2026b] catalog panel since 1996; [Hong et al. 2025] 2004 to 2019 syllabi.
- ExploreCourses accepts `academicYear=YYYYYYYY` back to 2010-11 (some archived pages now demand
  login; unverified for the XML endpoint).

### 2.7 Open datasets
Open Syllabus (restricted access), Course-Skill Atlas (public, aggregated), Light's panel (not public),
MOOCCube [Yu et al. 2020], UniSkill, Han et al.'s OSF deposit, a USC course catalog on Hugging Face.
No public, multi-institution, course-level description corpus with provenance exists, which makes the
Stanford raw snapshot worth archiving as a citable dataset (Zenodo DOI) rather than only a release asset.

## 3. Methods literature relevant to the pipeline

**2-d layouts.** Chari & Pachter [2023] argue that t-SNE/UMAP embeddings distort distances badly enough
to be misleading; Lause, Berens & Kobak [2024] reply that neighbourhood and class structure survive if
one measures the right things. Kobak & Berens [2019] give the three metrics that settle such arguments:
KNN (k-nearest-neighbour preservation, local), KNC (preservation of class-centroid nearest neighbours,
mesoscale), and CPD (Spearman correlation of pairwise distances, global). Espadoto et al. [2019] survey
trustworthiness, continuity, neighbourhood hit, Shepard goodness, and stress across 44 DR methods;
Wang et al. [2021] and Huang et al. [2022] compare t-SNE, UMAP, TriMap, and PaCMAP on local versus
global preservation. Clustering in the 2-d layout, as this pipeline does, is exactly the practice
Chari & Pachter attack, so the record should carry the Kobak-Berens triple, not trustworthiness alone.

**Clustering stability.** von Luxburg [2010] is the overview; Ben-Hur, Elisseeff & Guyon [2002] and
Monti et al. [2003] (consensus clustering) are the resampling recipes; Ballester & Penner [2021] and
Rüdiger et al. [2022] apply the same logic to topic models. Nothing specific to UMAP+HDBSCAN pipelines
turned up beyond single-cell tooling, which is itself a gap a short methods note could fill.

**Evaluating names.** Chang et al. [2009] (word and topic intrusion) and Doogan & Buntine [2021] on why
automated coherence is not interpretability; Stammbach et al. [2023] and Pham et al. [2024] (TopicGPT)
on LLMs as topic evaluators and generators, validated against human judgements; Piper & Wu [2025] on LLM
topic labels; Krumdick et al. [2025] on the limits of LLM-as-judge without human grounding.

**Seeing structure that is not there.** Wickham et al. [2010] lineups (already planned as v1.1);
Hullman, Resnick & Adar [2015] hypothetical outcome plots as a way to show seed-to-seed variation.

**Model selection.** MTEB [Muennighoff et al. 2023] clustering tasks are the public benchmark; the
preregistered kNN label agreement is a neighbourhood-hit statistic in Espadoto's terms.

## 4. Improvements, grouped

### 4.1 Methodological / record
1. **Add the Kobak-Berens triple to the record** (KNN, KNC over departments and schools, CPD) next to
   trustworthiness, and cite Chari & Pachter and Lause et al. in the methodology as the debate being
   answered. Cheap: everything needed is in `umap_coords.npz` and `embeddings.npz`.
2. **Region stability.** Bootstrap courses (or perturb the UMAP seed) and compute per-region
   co-assignment rates (Monti-style consensus). Report per layer; see 4.2 item 3 for the display use.
   The preregistration already says stability is reported, not ranked on.
3. **Name evaluation with humans in the loop.** A label-intrusion task on a sample of regions (four
   names, one from elsewhere, pick the intruder) is the Chang et al. protocol adapted to region names;
   50 items is an afternoon. If an LLM judge is used at all, ground it on that sample [Krumdick 2025].
4. **State the claimed-versus-practised caveat** [Han et al. 2023] and the descriptions-versus-syllabi
   evidence [Jiang & Pardos 2020] in the methodology: the map is of what the catalog says is taught.
5. **Global-structure check on n_neighbors.** Galaxy used n_neighbors=100 for a 1M-point map; this
   pipeline uses 15. KNC at 15 versus a larger setting (one run, reported not adopted) would document
   what mesoscale structure the chosen setting trades away.
6. **Cite the analogies.** Pardos & Nam's validation set (cross-listings, equivalencies, analogies) is
   the nearest thing to a standard for course embeddings; the collapsed cross-listings here make the
   first unusable as written, which is worth saying explicitly.

### 4.2 Features and colormaps
1. **Interdisciplinarity index per course.** Han et al.'s index rebuilt in embedding space: similarity
   to each department (or school) centroid, weighted by inter-centroid distance; or a Rao-Stirling
   diversity over the department mix of the course's 15 nearest neighbours. Content-only, no LLM.
2. **Cross-listing count and displacement.** Number of codes per course (Brint/Jacobs' observable) and
   the course's distance from its home-department centroid in embedding space. The two together show
   whether cross-listing tracks content position.
3. **Region confidence** from 4.1 item 2, as a colormap or as hover text, so a viewer can see which
   named regions are robust and which are seed artefacts.
4. **Prerequisite structure.** Parse "Prerequisite" clauses from descriptions into a has-prerequisite
   flag and, where codes resolve, in-degree and chain depth [Stavrinides & Zuev 2022]. Worth a count of
   how many Stanford descriptions carry such text before committing.
5. **Work-activity layer.** Embed O*NET DWA texts with the same instruction wrapper and score each course
   by max cosine per DWA family (the Course-Skill Atlas recipe), giving a "dominant work activity"
   colormap; optionally weight by Eloundou et al.'s task exposure for an AI-exposure colormap in the
   spirit of Light [2026a]. Provenance is clean (public O*NET, public ratings), but the methodology
   page would need to say the DWA alignment is a model inference.
6. **SDG tags** via the Kharlashkin et al. recipe. This is LLM enrichment, which v1 declined; it stays
   cheap if it ever becomes wanted.
7. **Research-frontier proximity** [Biasi & Ma 2022]: similarity of each description to recent versus
   older OpenAlex abstracts (Stanford-affiliated works would keep it institution-specific). An
   experiment rather than a v1.1 feature.
8. **Course-number level** as a finer colormap than career, motivated by Pardos & Nam's lower-division
   finding (4.3 item 1).
9. **First year offered**, if the back-year fetch in 4.4 item 1 happens.

### 4.3 Findings to validate independently on Stanford
1. **Lower-division courses are less disciplinarily structured** [Pardos & Nam 2020, from enrollment].
   Test: kNN department and school agreement by course-number level (or UG versus GR). Agreement
   should rise with level. Concordance between an enrollment signal at Berkeley and a content signal at
   Stanford would be the interesting result either way.
2. **Discipline similarity matrix** [Han et al. 2023]: sociology-anthropology 0.39, English-physics 0.01
   under tf-idf. Reproduce on Stanford departments mapped to their 20 disciplines, once with tf-idf
   (their method) and once with Qwen centroids; report the rank correlation between the two matrices.
3. **Engineering core, natural sciences peripheral** [Gim et al. 2025, Korea, course names]: centrality
   of Stanford departments in the department-centroid similarity graph the structure report already
   builds. Engineering-heavy cores would replicate; Stanford's H&S-heavy catalog may not.
4. **Cross-listing follows content proximity** [Jacobs 2014 reading]: department pairs that share
   cross-listed courses should have closer centroids than pairs that do not, controlling for school.
5. **Bag-of-words suffices for field classification** [Open Syllabus, 84% on 70 fields]: a linear probe
   on the Qwen embeddings for school and department, versus a tf-idf probe, gives the comparable number
   and quantifies how much the org chart is recoverable from content alone.
6. **Disciplinary boundaries are stable over time** [Hong et al. 2025] and **task exposure barely moved
   in 15 years** [Light 2026a]: both need back-year catalogs (4.4 item 1).

### 4.4 Other
1. **Longitudinal Stanford.** Two sources: ExploreCourses back-years via `academicYear` (2010-11 on,
   endpoint access unverified) and UC-CHP's digitised Stanford courses 1900 to 2011. Aligning courses
   across years by code and title, embedding once, and animating the layout by decade is a distinct
   second map and the most productive extension the literature points to.
2. **Archive the snapshot as a dataset** with a DOI; the Course-Skill Atlas paper is the template for
   describing coverage and limitations.
3. **Second institution.** Han et al.'s 80-school catalog corpus and Light's panel show catalogs scrape
   at scale; MIT (FireRoad) remains the cheap next adapter. Cross-institution alignment is where the
   articulation literature [Pardos et al. 2019; Kim et al. 2024] becomes relevant.
4. **Readers.** The Pardos lab (Berkeley, AskOski), Kizilcec lab (Cornell), Stanford's Pathways Lab /
   Carta, and Morgan Frank's group (Pittsburgh, Course-Skill Atlas) are the audiences most likely to
   cite or reuse the map and record.

## References

- Aasheim, C. L., Williams, S., Rutner, P., & Gardiner, A. (2015). Data analytics vs. data science: a study of similarities and differences in undergraduate programs based on course descriptions. *JAIS*. https://openalex.org/W2505749379
- Abbott, A. (2001). *Chaos of Disciplines*. University of Chicago Press.
- Aldrich, P. R. (2015). The curriculum prerequisite network: modeling the curriculum as a complex system. *Biochemistry and Molecular Biology Education*. https://doi.org/10.1002/bmb.20861
- Ballester, O., & Penner, O. (2021). Robustness, replicability and scalability in topic modelling. *Journal of Informetrics*. https://doi.org/10.1016/j.joi.2021.101224
- Ben-Hur, A., Elisseeff, A., & Guyon, I. (2002). A stability based method for discovering structure in clustered data. *Pacific Symposium on Biocomputing*. https://doi.org/10.1142/9789812799623_0002
- Biasi, B., & Ma, S. (2022). The education-innovation gap. NBER Working Paper 29853. https://doi.org/10.3386/w29853 (revised August 2026 as *Frontier knowledge in higher education*; the gap definition above is from the May 2023 version, SSRN 4072258)
- Bleemer, Z. (2018). The UC ClioMetric History Project and formatted optical character recognition. CSHE 3.18. https://escholarship.org/uc/item/9xz1748q ; data at https://uccliometric.org/courses/
- Börner, K., et al. (2012). Design and update of a classification system: the UCSD map of science. *PLoS ONE*. https://doi.org/10.1371/journal.pone.0039464
- Börner, K., Scrivner, O., Gallant, M., et al. (2018). Skill discrepancies between research, education, and jobs reveal the critical need to supply soft skills for the data economy. *PNAS*. https://doi.org/10.1073/pnas.1804247115
- Brint, S., Turk-Bicakci, L., Proctor, K., & Murphy, S. P. (2008). Expanding the social frame of knowledge: interdisciplinary, degree-granting fields in American colleges and universities, 1975-2000. *Review of Higher Education*. https://doi.org/10.1353/rhe.0.0042
- Buzzell, A., Atherton, T. J., & Barthelemy, R. S. (2025). Modern physics courses: understanding the content taught in the U.S. *Phys. Rev. PER*. https://doi.org/10.1103/physrevphyseducres.21.010139
- Chang, J., Gerrish, S., Wang, C., Boyd-Graber, J., & Blei, D. (2009). Reading tea leaves: how humans interpret topic models. *NeurIPS*. https://openalex.org/W2159426623
- Chari, T., & Pachter, L. (2023). The specious art of single-cell genomics. *PLoS Computational Biology*. https://doi.org/10.1371/journal.pcbi.1011288
- Chaturapruek, S., Dalberg, T., Thompson, M. E., Giebel, S., Harrison, M. H., Johari, R., Stevens, M. L., & Kizilcec, R. F. (2021). Studying undergraduate course consideration at scale. *AERA Open*. https://doi.org/10.1177/2332858421991148
- Chau, H., Bana, S. H., Bouvier, B., & Frank, M. R. (2023). Connecting higher education to workplace activities and earnings. *PLoS ONE*. https://doi.org/10.1371/journal.pone.0282323
- Chau, H., Yu, R., Pardos, Z., et al. (2025). Skill-based explanations for serendipitous course recommendation. arXiv:2508.19569 (abstract).
- Chen, Y., Fu, A., Lee, J. J., et al. (2022). Pathways: exploring academic interests with historical course enrollment records. *L@S*. https://doi.org/10.1145/3491140.3528270
- Chen, Y., Iyer, P., & Kizilcec, R. F. (2024). Reflection on purpose changes students' academic interests: a scalable intervention in an online course catalog. arXiv:2412.19035
- Dalberg, T., Cortes, K. E., & Stevens, M. L. (2024). Major selection as iteration. *AERA Open*. https://doi.org/10.1177/23328584241249600
- Dong, M., Yu, R., & Pardos, Z. A. (2019). Design and deployment of a better course search tool: inferring latent keywords from enrollment networks. *EC-TEL*. https://doi.org/10.1007/978-3-030-29736-7_36
- Doogan, C., & Buntine, W. (2021). Topic model or topic twaddle? Re-evaluating semantic interpretability measures. *NAACL*. https://doi.org/10.18653/v1/2021.naacl-main.300
- Eloundou, T., Manning, S., Mishkin, P., & Rock, D. (2023). GPTs are GPTs. arXiv:2303.10130
- Espadoto, M., Martins, R. M., Kerren, A., Hirata, N. S. T., & Telea, A. C. (2019). Toward a quantitative survey of dimension reduction techniques. *IEEE TVCG*. https://doi.org/10.1109/tvcg.2019.2944182
- Frank, D. J., & Gabler, J. (2006). *Reconstructing the University: Worldwide Shifts in Academia in the 20th Century*. Stanford University Press.
- Gim, G., Yun, J., & Lee, S. H. (2025). Quantifying interdisciplinary synergy in higher STEM education. *EPJ Data Science*. https://doi.org/10.1140/epjds/s13688-025-00566-6
- Grunspan, D. Z., et al. (2019). The state of evolutionary medicine in undergraduate education. *Evolution, Medicine, and Public Health*. https://doi.org/10.1093/emph/eoz012
- Han, S., LaViolette, J., Borkenhagen, C., McAllister, W., & Bearman, P. S. (2023). Interdisciplinary college curriculum and its labor market implications. *PNAS*. https://doi.org/10.1073/pnas.2221915120 ; data https://doi.org/10.17605/OSF.IO/5A9WK
- Heileman, G. L., Abdallah, C. T., Slim, A., & Hickman, M. (2018). Curricular analytics: a framework for quantifying the impact of curricular reforms and pedagogical innovations. arXiv:1811.09676
- Herzog, P. S., Ai, J., & Ashton, J. W. (2022). Applying bibliometric techniques: studying interdisciplinarity in higher education curriculum. *Computation*. https://doi.org/10.3390/computation10020026
- Hong, Y., Kim, B., Jeon, J., & Kim, L. (2025). Has higher education become more interdisciplinary? A longitudinal analysis of syllabi using natural language processing. *Humanities and Social Sciences Communications*. https://doi.org/10.1057/s41599-025-06126-7
- Huang, H., Wang, Y., Rudin, C., & Browne, E. P. (2022). Towards a comprehensive evaluation of dimension reduction methods for transcriptomic data visualization. *Communications Biology*. https://doi.org/10.1038/s42003-022-03628-x
- Hullman, J., Resnick, P., & Adar, E. (2015). Hypothetical outcome plots outperform error bars and violin plots. *PLoS ONE*. https://doi.org/10.1371/journal.pone.0142444
- Jacobs, J. A. (2014). *In Defense of Disciplines*. University of Chicago Press.
- Javadian Sabet, A., Bana, S. H., Yu, R., & Frank, M. R. (2024). Course-Skill Atlas: a national longitudinal dataset of skills taught in U.S. higher education curricula. *Scientific Data*. https://doi.org/10.1038/s41597-024-03931-8
- Jiang, W., & Pardos, Z. A. (2020). Evaluating sources of course information and models of representation on a variety of institutional prediction tasks. *EDM*. https://eric.ed.gov/?id=ED607904
- Kharlashkin, L., Macías, M. M., Huovinen, L., & Hämäläinen, M. (2024). Predicting Sustainable Development Goals using course descriptions: from LLMs to conventional foundation models. *JDMDH*. https://doi.org/10.46298/jdmdh.13127
- Kim, M., Puder, A., Hayward, C., et al. (2024). Foundation models for course equivalency evaluation. *ICDM Workshops*. https://doi.org/10.1109/ICDMW65004.2024.00045
- Klavans, R., & Boyack, K. W. (2017). Which type of citation analysis generates the most accurate taxonomy of scientific and technical knowledge? *JASIST*. https://doi.org/10.1002/asi.23734
- Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell transcriptomics. *Nature Communications*. https://doi.org/10.1038/s41467-019-13056-x
- Krumdick, M., Lovering, C., Reddy, V., et al. (2025). No free labels: limitations of LLM-as-a-judge without human grounding. arXiv:2503.05061
- Kwak, Y., Schmucker, R., & Pardos, Z. A. (2026). Strengthening course transfer pathways using graph-theoretic articulation networks. *LAK*. https://doi.org/10.1145/3785022.3785107 (abstract)
- Lang, D., Wang, A., Dalal, N., Paepcke, A., & Stevens, M. L. (2022). Forecasting undergraduate majors: a natural language approach. *AERA Open*. https://doi.org/10.1177/23328584221126516
- Lause, J., Berens, P., & Kobak, D. (2024). The art of seeing the elephant in the room: 2D embeddings of single-cell data do make sense. *PLoS Computational Biology*. https://doi.org/10.1371/journal.pcbi.1012403
- Liang, C., Ye, J., Wu, Z., Pursel, B., & Giles, C. L. (2017). Recovering concept prerequisite relations from university course dependencies. *AAAI*. https://doi.org/10.1609/aaai.v31i1.10550
- Light, J. (2026a). How exposed is higher education to artificial intelligence? Working paper. https://jacob-light.github.io/field-ai-exposure.pdf
- Light, J. (2026b). Student demand and the supply of college courses. Working paper (3 Aug 2026). https://jacob-light.github.io/catalog-project.pdf
- McClure, D. (2021). Galaxy v2. Open Syllabus blog. https://blog.opensyllabus.org/galaxy-v2 ; viewer https://galaxy.opensyllabus.org/ ; methods https://docs.opensyllabus.org/methods.html
- Monti, S., Tamayo, P., Mesirov, J., & Golub, T. (2003). Consensus clustering. *Machine Learning*. https://doi.org/10.1023/a:1023949509487
- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. *EACL*. https://doi.org/10.18653/v1/2023.eacl-main.148
- Musazade, N., Mezei, J., & Zhang, M. (2026). UniSkill: a dataset for matching university curricula to professional competencies. arXiv:2603.03134
- Pardos, Z. A., Chau, H., & Zhao, H. (2019). Data-assistive course-to-course articulation using machine translation. *L@S*. https://doi.org/10.1145/3330430.3333622
- Pardos, Z. A., & Nam, A. J. H. (2020). A university map of course knowledge. *PLoS ONE*. https://doi.org/10.1371/journal.pone.0233207
- Pham, C. M., Hoyle, A., Sun, S., Resnik, P., & Iyyer, M. (2024). TopicGPT: a prompt-based topic modeling framework. *NAACL*. https://doi.org/10.18653/v1/2024.naacl-long.164
- Piper, A., & Wu, S. (2025). Evaluating large language models for narrative topic labeling. *NLP4DH*. https://doi.org/10.18653/v1/2025.nlp4dh-1.25
- Prior, D. D., Mysore Seshadrinath, S., Zhang, M., et al. (2024). Measuring sustainable development goals in higher education through semantic matching. *Studies in Higher Education*. https://doi.org/10.1080/03075079.2024.2386625
- Ren, D., Hohman, F., Lin, H., et al. (2025). Embedding Atlas: low-friction, interactive embedding visualization. *IEEE VIS*. https://doi.org/10.1109/vis60296.2025.00044
- Rüdiger, M., Antons, D., Joshi, A. M., & Salge, T.-O. (2022). Topic modeling revisited: new evidence on algorithm performance and quality metrics. *PLoS ONE*. https://doi.org/10.1371/journal.pone.0266325
- Stammbach, D., Zouhar, V., Hoyle, A., Sachan, M., & Ash, E. (2023). Revisiting automated topic model evaluation with large language models. *EMNLP*. https://doi.org/10.18653/v1/2023.emnlp-main.581
- Stavrinides, P., & Zuev, K. M. (2022). Course-prerequisite networks for analyzing and understanding academic curricula. *Applied Network Science*. https://doi.org/10.1007/s41109-023-00543-w
- Suominen, A., & Toivanen, H. (2015). Map of science with topic modeling: comparison of unsupervised learning and human-assigned subject classification. *JASIST*. https://doi.org/10.1002/asi.23596
- van Eck, N. J., & Waltman, L. (2010). Software survey: VOSviewer. *Scientometrics*. https://doi.org/10.1007/s11192-009-0146-3
- von Luxburg, U. (2010). Clustering stability: an overview. *Foundations and Trends in Machine Learning*. https://doi.org/10.1561/2200000008
- Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). Understanding how dimension reduction tools work: an empirical approach to deciphering t-SNE, UMAP, TriMap, and PaCMAP. *JMLR* 22(201):1-73. https://jmlr.org/papers/v22/20-1061.html (arXiv:2012.04456)
- Wickham, H., Cook, D., Hofmann, H., & Buja, A. (2010). Graphical inference for infovis. *IEEE TVCG*. https://doi.org/10.1109/tvcg.2010.161
- Xiao, K., Bai, Y., & Wang, Z. (2022). Extracting prerequisite relations among concepts from the course descriptions. *IJSEKE*. https://doi.org/10.1142/s0218194022400034 (abstract)
- Xu, Y., & Pardos, Z. A. (2024). Extracting course similarity signal using subword embeddings. *LAK*. https://doi.org/10.1145/3636555.3636903
- Xu, Z., Li, X., Huan, Y., et al. (2025). From course to skill: evaluating LLM performance in curricular analytics. arXiv:2505.02324
- Xu, Z., Guan, X., Shi, C., et al. (2026). Evaluating 21st-century competencies in postsecondary curricula with large language models. *Journal of Learning Analytics*. arXiv:2601.10983 (abstract)
- Yang, B., Gharebhaygloo, M., Rondi, H. R., et al. (2024). Comparative analysis of course prerequisite networks for five Midwestern public institutions. *Applied Network Science*. https://doi.org/10.1007/s41109-024-00637-z
- Yu, J., Luo, G., Xiao, T., et al. (2020). MOOCCube: a large-scale data repository for NLP applications in MOOCs. *ACL*. https://doi.org/10.18653/v1/2020.acl-main.285
- Yu, R., Das, S., Gurajada, S., et al. (2021). A research framework for understanding education-occupation alignment with NLP techniques. *NLP4PosImpact*. https://doi.org/10.18653/v1/2021.nlp4posimpact-1.11
- Yu, T., Qi, Y., & Zhang, T. (2024). How to measure the interdisciplinarity of library and information science courses? *JOLIS*. https://doi.org/10.1177/09610006231222627 (abstract)

Data from OpenAlex (CC0) and Semantic Scholar.
