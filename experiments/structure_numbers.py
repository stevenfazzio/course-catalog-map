"""Post-hoc structure analysis, step 2: how much do the models' native (EVoC) hierarchies agree with each
other, with the catalog's own structure, and with their 2-d (ToponymyClusterer on UMAP) partitions?

    uv run python experiments/structure_numbers.py

Adjusted mutual information (AMI) between partitions, computed on courses that both partitions cluster
(EVoC and the 2-d clusterer both leave a noise set). Layers are matched by nearest region count.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami
from toponymy import ToponymyClusterer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

MODELS = ["qwen3-0.6b", "qwen3-4b", "arctic-l-v2", "nomic-v2-moe"]
TARGET_REGIONS = [40, 120]  # coarse and fine comparison points


def load_evoc(paths, key):
    files = config.keyed_files(paths, key)
    d = np.load(files["npz"].with_name(files["npz"].name.replace("embeddings", "evoc_layers")), allow_pickle=True)
    return [d[k] for k in sorted((k for k in d.files if k.startswith("layer_")), key=lambda k: int(k.split("_")[1]))]


def layout_layers(paths, key):
    files = config.keyed_files(paths, key)
    coords = np.load(files["umap"], allow_pickle=True)["coords"]
    emb = np.load(files["npz"], allow_pickle=True)["embeddings"]
    cl = ToponymyClusterer(min_clusters=6, base_min_cluster_size=10, verbose=False, show_progress_bar=False)
    cl.fit(clusterable_vectors=coords, embedding_vectors=emb)
    return [np.asarray(layer.cluster_labels) for layer in cl.cluster_layers_]


def pick(layers, target):
    return min(layers, key=lambda lab: abs(lab.max() + 1 - target))


def ami_clustered(a, b):
    m = (a >= 0) & (b >= 0)
    return ami(a[m], b[m]), float(m.mean())


def main():
    paths = config.source_paths(config.DEFAULT_SOURCE)
    corpus = pd.read_parquet(paths["corpus"])
    dept = pd.factorize(corpus["department_code"])[0]
    school = pd.factorize(corpus["school"])[0]
    evoc = {k: load_evoc(paths, k) for k in MODELS}
    twod = {k: layout_layers(paths, k) for k in MODELS}

    for target in TARGET_REGIONS:
        print(f"\n── layers nearest {target} regions ──")
        print(
            f"{'model':14} {'EVoC regions':>12} {'clustered':>9} {'AMI dept':>8} {'AMI school':>10} | {'2-d regions':>11} {'clustered':>9} {'AMI dept':>8} {'AMI school':>10} | {'AMI EVoC vs 2-d':>15}"
        )
        for k in MODELS:
            e = pick(evoc[k], target)
            t = pick(twod[k], target)
            ed, _ = ami_clustered(e, dept)
            es, _ = ami_clustered(e, school)
            td, _ = ami_clustered(t, dept)
            ts, _ = ami_clustered(t, school)
            et, cov = ami_clustered(e, t)
            print(
                f"{k:14} {e.max() + 1:12d} {(e >= 0).mean():9.0%} {ed:8.3f} {es:10.3f} | {t.max() + 1:11d} {(t >= 0).mean():9.0%} {td:8.3f} {ts:10.3f} | {et:15.3f}"
            )
        print("  AMI between models' EVoC layers (courses clustered by both):")
        for i, a in enumerate(MODELS):
            row = []
            for b in MODELS:
                v, cov = ami_clustered(pick(evoc[a], target), pick(evoc[b], target))
                row.append(f"{v:.3f}({cov:.0%})")
            print(f"    {a:14} " + "  ".join(row))


if __name__ == "__main__":
    main()
