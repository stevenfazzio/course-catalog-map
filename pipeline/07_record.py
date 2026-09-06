"""Stage 07: copy the small artifacts that make the map auditable into docs/record/<source>/, which is versioned.

data/ is gitignored and the named regions cannot be regenerated bit-identically, so the files that say what the
published map actually contains (labels, names, tree, corpus rules and drops, run metadata, the comparison
results, the cleaned corpus itself) are copied here and committed. The raw catalog XML and the embeddings are
too large for git and belong in a release asset.
"""

import argparse
import shutil

import config

RECORD_FILES = [
    "corpus.parquet",
    "corpus_meta.json",
    "corpus_drops.csv",
    "drops.csv",
    "embeddings_meta.json",
    "umap_meta.json",
    "labels.parquet",
    "topic_names.json",
    "cluster_tree.json",
    "labels_meta.json",
    "embedding_comparison.json",
    "embedding_comparison_raw.json",
    # step 1 of docs/plan.md (experiments/record_metrics.py): layout metrics and region stability, names off
    "layout_metrics.json",
    "stability.parquet",
    "stability_regions.csv",
    "stability_meta.json",
    "stability_layouts.npz",
    # step 2 (experiments/wayfinding_lineup.py): can a listener find a region from its name among its neighbours
    "wayfinding_items.json",
    "wayfinding_lineup.csv",
    "wayfinding_lineup.json",
    "wayfinding_calls.jsonl",  # the listener's raw answers
    "wayfinding_calls.runs.json",  # what the calls cost, per paid run
    # step 3 (experiments/name_intrusion.py): the human name-intrusion task, its sealed key, answers and result
    "name_intrusion_items.csv",
    "name_intrusion_key.json",
    "name_intrusion_rater.html",
    "name_intrusion_answers.csv",
    "name_intrusion_llm.csv",
    "name_intrusion_calls.jsonl",
    "name_intrusion_calls.runs.json",
    "name_intrusion.json",
    # step 4 (experiments/level_agreement.py, crosslisting_distance.py, linear_probe.py, department_centrality.py):
    # validation checks against findings in the literature, names off
    "validation_level.json",
    "validation_crosslisting.json",
    "validation_crosslisting_pairs.csv",
    "crosslisting_displacement.parquet",
    "validation_probe.json",
    "validation_probe_predictions.parquet",
    "validation_probe_departments.csv",
    "validation_centrality.json",
    "validation_centrality.csv",
]
RECORD_GLOBS = ["evoc_topic_names*.json", "evoc_labels_meta*.json", "name_guardrail*.csv", "raw/_manifest.json"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    dest = config.DOCS_DIR / "record" / args.source
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in RECORD_FILES:
        src = paths["base"] / name
        if src.exists():
            shutil.copy2(src, dest / name)
            copied.append(name)
    for pattern in RECORD_GLOBS:
        for src in sorted(paths["base"].glob(pattern)):
            target = dest / src.name
            shutil.copy2(src, target)
            copied.append(src.name)
    total = sum((dest / n).stat().st_size for n in copied) / 1e6
    print(f"{len(copied)} files, {total:.1f} MB -> {dest}")
    for n in copied:
        print("  ", n)


if __name__ == "__main__":
    main()
