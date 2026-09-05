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
