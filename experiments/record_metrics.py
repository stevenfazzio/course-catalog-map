"""Step 1 of docs/plan.md: layout-quality metrics and region stability for the published map, names off.

    OMP_NUM_THREADS=1 uv run python experiments/record_metrics.py [--source stanford] [--quick]

Writes to data/<source>/ (pipeline/07_record.py copies them into docs/record/<source>/):

    layout_metrics.json     the Kobak-Berens triple (KNN, KNC over departments and schools, CPD) and trustworthiness
                            for the published layout; the same for one layout at a larger n_neighbors (reported, not
                            adopted) and for each seed layout
    stability.parquet       one row per course, aligned to corpus.parquet: for every published layer, the share of its
                            published-region peers the course stays with, averaged over the seed runs and over the
                            subsample runs
    stability_regions.csv   one row per published region: mean pairwise co-assignment of its members over the seed runs
                            and over the subsample runs
    stability_meta.json     run record: parameters, per-run layer counts and layer matching, per-layer AMI against the
                            published clustering, the local re-clustering control, versions, timings
    stability_layouts.npz   every rerun's coordinates and cluster layers, so step 2 (lineups) and the display options in
                            step 6 need no new UMAP runs

Two perturbations, Toponymy clustering only (no LLM): UMAP at ten other seeds on the full corpus, and ten 80%
subsamples of the corpus at the published seed. The reference is the published labels.parquet, not a local
re-clustering, because the published regions are what a viewer sees. The clusterer is deterministic on one machine but
not across machines (CLAUDE.md), so a control re-clusters the published layout here and reports its agreement with the
published labels: the share of the disagreement that is the clusterer's machine-to-machine variation rather than the
perturbation.

Definitions. A pair of courses from one published region is co-assigned in a rerun when both land in the same rerun
region; two courses the rerun leaves unclustered are not co-assigned. Per course, "stays with" counts the peers of its
published region that share its rerun region, so an unclustered course stays with none of them. Subsample statistics
use only pairs and courses present in the subsample. Published layers are matched to rerun layers by nearest region
count. AMI is computed on courses that both partitions cluster, with the share of published members the rerun leaves
unclustered reported next to it.
"""

import argparse
import datetime as dt
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score
from toponymy import ToponymyClusterer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import compare_embeddings as ce  # noqa: E402
import config  # noqa: E402
from io_utils import atomic_write_text, write_parquet_safely  # noqa: E402

SEEDS = list(range(43, 53))  # ten seeds; 42 is the published layout
SUBSAMPLE_FRACTION = 0.8
SUBSAMPLE_RNG_SEEDS = list(range(10))
LARGER_N_NEIGHBORS = 100  # Open Syllabus Galaxy's setting (lit_review 4.1.5); reported, not adopted
KNN_KS = (10, 15)  # 10 is Kobak & Berens's; 15 is the pipeline's n_neighbors and the preregistered k
KNC_K_DEPARTMENT = 10
KNC_K_SCHOOL = 3  # nine schools
KNC_MIN_DEPARTMENT_SIZE = 20  # a second KNC row on departments large enough for a steady centroid
CPD_SAMPLE = 1_000
CPD_DRAWS = 5
TRUST_K = 15


# ── metrics ──────────────────────────────────────────────────────────────────


def neighbour_overlap(neigh_a: np.ndarray, neigh_b: np.ndarray) -> float:
    """Mean share of each row's neighbours in `neigh_a` that also appear in its row of `neigh_b`."""
    assert neigh_a.shape == neigh_b.shape
    k = neigh_a.shape[1]
    return float(np.mean([len(set(a) & set(b)) for a, b in zip(neigh_a, neigh_b)]) / k)


def knn_preservation(neigh_hd: np.ndarray, Y: np.ndarray) -> float:
    """Kobak & Berens KNN: share of each point's k nearest neighbours in the embedding space (cosine, precomputed)
    that are also among its k nearest in the layout (euclidean)."""
    k = neigh_hd.shape[1]
    return neighbour_overlap(neigh_hd, ce.knn_indices(Y, k, metric="euclidean"))


def class_centroids(X: np.ndarray, classes: np.ndarray, min_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector per class, for classes with at least `min_size` members. Returns (class values, centroids)."""
    values, inverse, counts = np.unique(classes, return_inverse=True, return_counts=True)
    sums = np.zeros((len(values), X.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, X)
    keep = counts >= min_size
    return values[keep], sums[keep] / counts[keep, None]


def knc_preservation(X: np.ndarray, Y: np.ndarray, classes: np.ndarray, k: int, min_size: int = 1) -> dict:
    """Kobak & Berens KNC: share of each class centroid's k nearest class centroids in the embedding space (cosine)
    that are also among its k nearest in the layout (euclidean)."""
    values, cx = class_centroids(X, classes, min_size)
    _, cy = class_centroids(Y, classes, min_size)
    assert k < len(values), f"k={k} needs more than {len(values)} classes"
    nx = ce.knn_indices(cx, k, metric="cosine")
    ny = ce.knn_indices(cy, k, metric="euclidean")
    return {"value": neighbour_overlap(nx, ny), "k": k, "n_classes": int(len(values))}


def cpd(X: np.ndarray, Y: np.ndarray, n_sample: int = CPD_SAMPLE, seed: int = 0) -> float:
    """Kobak & Berens CPD: Spearman correlation between pairwise distances in the embedding space (cosine) and in
    the layout (euclidean) over a random sample of points."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    return float(spearmanr(pdist(X[idx], "cosine"), pdist(Y[idx], "euclidean")).statistic)


def match_layers(published_counts: list[int], rerun_counts: list[int]) -> list[int]:
    """For each published layer, the index of the rerun layer with the nearest region count."""
    return [int(np.argmin([abs(c - p) for c in rerun_counts])) for p in published_counts]


def region_agreement(published: np.ndarray, rerun: np.ndarray, present: np.ndarray | None = None):
    """How a rerun treats each published region.

    Returns (coassign, stay): `coassign[r]` is the share of pairs of present members of published region r that share a
    rerun region; `stay[i]` is the share of course i's present published-region peers that share its rerun region. A
    course the rerun leaves unclustered is co-assigned with nobody and stays with nobody. NaN where undefined: courses
    unlabelled in the published layer, courses not present, regions with fewer than two present members.
    """
    n = len(published)
    present = np.ones(n, dtype=bool) if present is None else present
    stay = np.full(n, np.nan)
    coassign: dict[int, float] = {}
    for r in np.unique(published[published >= 0]):
        members = np.flatnonzero((published == r) & present)
        m = len(members)
        if m < 2:
            coassign[int(r)] = np.nan
            continue
        q = rerun[members]
        clustered = q >= 0
        _, inverse, counts = np.unique(q[clustered], return_inverse=True, return_counts=True)
        same = np.zeros(m)
        same[clustered] = counts[inverse] - 1  # peers sharing the course's rerun region
        stay[members] = same / (m - 1)
        coassign[int(r)] = float((counts * (counts - 1) / 2).sum() / (m * (m - 1) / 2))
    return coassign, stay


def layer_agreement(published: np.ndarray, rerun: np.ndarray, present: np.ndarray | None = None) -> dict:
    """AMI between a published layer and its matched rerun layer on courses both cluster, with coverage."""
    present = np.ones(len(published), dtype=bool) if present is None else present
    both = (published >= 0) & (rerun >= 0) & present
    pub_clustered = (published >= 0) & present
    return {
        "ami": float(adjusted_mutual_info_score(published[both], rerun[both])),
        "share_both_clustered": float(both.sum() / present.sum()),
        "published_members_unclustered": float(((rerun < 0) & pub_clustered).sum() / pub_clustered.sum()),
        "n_clusters": int(rerun.max() + 1),
        "unlabelled_share": float((rerun[present] < 0).mean()),
    }


# ── runs ─────────────────────────────────────────────────────────────────────


def umap_layout(X: np.ndarray, seed: int, n_neighbors: int = config.UMAP_N_NEIGHBORS) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        n_components=2,
        random_state=seed,
    )
    return reducer.fit_transform(X).astype(np.float32)


def cluster_layers(coords: np.ndarray, embeddings: np.ndarray) -> list[np.ndarray]:
    """Stage 04's clusterer, same settings, no namer. Layer 0 is the finest."""
    clusterer = ToponymyClusterer(
        min_clusters=config.TOPONYMY_MIN_CLUSTERS,
        base_min_cluster_size=config.TOPONYMY_BASE_MIN_CLUSTER_SIZE,
        verbose=False,
        show_progress_bar=False,
    )
    np.random.seed(config.UMAP_RANDOM_STATE)
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=embeddings)
    return [np.asarray(layer.cluster_labels, dtype=np.int32) for layer in clusterer.cluster_layers_]


def layout_metrics(X: np.ndarray, Y: np.ndarray, neigh_hd: dict[int, np.ndarray], classes: dict) -> dict:
    out = {f"knn_k{k}": knn_preservation(neigh_hd[k], Y) for k in KNN_KS}
    out["knc_department"] = knc_preservation(X, Y, classes["department"], KNC_K_DEPARTMENT)
    out[f"knc_department_min{KNC_MIN_DEPARTMENT_SIZE}"] = knc_preservation(
        X, Y, classes["department"], KNC_K_DEPARTMENT, min_size=KNC_MIN_DEPARTMENT_SIZE
    )
    out["knc_school"] = knc_preservation(X, Y, classes["school"], KNC_K_SCHOOL)
    draws = [cpd(X, Y, seed=s) for s in range(CPD_DRAWS)]
    out["cpd"] = {"mean": float(np.mean(draws)), "min": float(np.min(draws)), "max": float(np.max(draws))}
    return out


def expand(values: np.ndarray, index: np.ndarray, n: int, fill) -> np.ndarray:
    full = np.full(n, fill, dtype=values.dtype)
    full[index] = values
    return full


def compare_run(published: list[np.ndarray], rerun: list[np.ndarray], present: np.ndarray | None = None) -> dict:
    """Match layers, then per-layer agreement, per-region co-assignment and per-course stay for one rerun."""
    matched = match_layers([int(p.max()) + 1 for p in published], [int(r.max()) + 1 for r in rerun])
    out = {"matched_layers": matched, "rerun_counts": [int(r.max()) + 1 for r in rerun], "layers": []}
    for pub, j in zip(published, matched):
        coassign, stay = region_agreement(pub, rerun[j], present)
        out["layers"].append({"agreement": layer_agreement(pub, rerun[j], present), "coassign": coassign, "stay": stay})
    return out


def summarise(runs: list[dict], published: list[np.ndarray], n: int) -> tuple[list[dict], list[dict], np.ndarray]:
    """Average the per-run results: per layer (AMI etc.), per region (co-assignment), per course (stay)."""
    per_layer, per_region = [], []
    stay = np.full((len(published), n), np.nan)
    for i, pub in enumerate(published):
        agreements = [run["layers"][i]["agreement"] for run in runs]
        per_layer.append(
            {
                "published_n_clusters": int(pub.max()) + 1,
                "matched_rerun_n_clusters": [run["rerun_counts"][run["matched_layers"][i]] for run in runs],
                **{
                    key: {
                        "mean": float(np.mean([a[key] for a in agreements])),
                        "min": float(np.min([a[key] for a in agreements])),
                        "max": float(np.max([a[key] for a in agreements])),
                    }
                    for key in ("ami", "published_members_unclustered", "unlabelled_share")
                },
            }
        )
        for r in range(int(pub.max()) + 1):
            vals = np.array([run["layers"][i]["coassign"].get(r, np.nan) for run in runs])
            per_region.append(
                {
                    "layer": i,
                    "cluster": r,
                    "n": int((pub == r).sum()),
                    "coassign_mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
                    "coassign_min": float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan,
                    "n_runs": int(np.isfinite(vals).sum()),
                }
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # nanmean of an all-NaN column (never present) is NaN
            stay[i] = np.nanmean(np.vstack([run["layers"][i]["stay"] for run in runs]), axis=0)
    return per_layer, per_region, stay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--quick", action="store_true", help="two seeds, two subsamples, no larger-n_neighbors run")
    args = ap.parse_args()
    seeds = SEEDS[:2] if args.quick else SEEDS
    sub_seeds = SUBSAMPLE_RNG_SEEDS[:2] if args.quick else SUBSAMPLE_RNG_SEEDS
    paths = config.source_paths(args.source)
    files = config.keyed_files(paths)
    t_start = time.time()

    corpus = pd.read_parquet(paths["corpus"])
    ids = corpus["course_id"].to_numpy()
    emb = np.load(files["npz"], allow_pickle=True)
    lay = np.load(files["umap"], allow_pickle=True)
    labels = pd.read_parquet(files["labels"])
    assert (emb["course_id"] == ids).all() and (lay["course_id"] == ids).all() and (labels["course_id"] == ids).all()
    X, Y0 = emb["embeddings"], lay["coords"]
    n = len(ids)
    layer_cols = sorted((c for c in labels.columns if c.startswith("cluster_layer_")), key=lambda c: int(c[14:]))
    published = [labels[c].to_numpy().astype(np.int32) for c in layer_cols]
    names = [labels[f"label_layer_{i}"].to_numpy() for i in range(len(published))]
    classes = {"department": corpus["department_code"].to_numpy(), "school": corpus["school"].to_numpy()}
    print(f"{n} courses, {len(published)} published layers: {[int(p.max()) + 1 for p in published]}")

    # ── layout metrics: published layout, larger n_neighbors, seeds ──
    t0 = time.time()
    neigh15 = ce.knn_indices(X, max(KNN_KS), metric="cosine")  # kneighbors returns rows sorted by distance
    neigh_hd = {k: neigh15[:, :k] for k in KNN_KS}
    print(f"embedding-space neighbours in {time.time() - t0:.0f}s")
    metrics = {"published": layout_metrics(X, Y0, neigh_hd, classes)}
    metrics["published"]["trustworthiness"] = {"value": ce.trustworthiness(X, Y0, TRUST_K), "k": TRUST_K}
    print(f"published layout: {json.dumps(metrics['published'])}")

    saved: dict[str, np.ndarray] = {"course_id": ids}
    if not args.quick:
        t0 = time.time()
        Y_big = umap_layout(X, config.UMAP_RANDOM_STATE, n_neighbors=LARGER_N_NEIGHBORS)
        big_layers = cluster_layers(Y_big, X)
        metrics[f"n_neighbors_{LARGER_N_NEIGHBORS}"] = layout_metrics(X, Y_big, neigh_hd, classes) | {
            "trustworthiness": {"value": ce.trustworthiness(X, Y_big, TRUST_K), "k": TRUST_K},
            "cluster_layer_counts": [int(lab.max()) + 1 for lab in big_layers],
            "seconds": round(time.time() - t0),
        }
        saved["nn100_coords"] = Y_big
        for j, lab in enumerate(big_layers):
            saved[f"nn100_layer_{j}"] = lab
        print(f"n_neighbors={LARGER_N_NEIGHBORS}: {json.dumps(metrics[f'n_neighbors_{LARGER_N_NEIGHBORS}'])}")

    # ── control: the published layout re-clustered on this machine ──
    control_a, control_b = cluster_layers(Y0, X), cluster_layers(Y0, X)
    deterministic = len(control_a) == len(control_b) and all(np.array_equal(a, b) for a, b in zip(control_a, control_b))
    control = compare_run(published, control_a)
    for j, lab in enumerate(control_a):
        saved[f"control_layer_{j}"] = lab
    print(
        f"control (published layout, local clusterer): layers {control['rerun_counts']}, deterministic here: "
        f"{deterministic}, AMI per published layer {[round(L['agreement']['ami'], 4) for L in control['layers']]}"
    )

    # ── seeds ──
    seed_runs = []
    metrics["seeds"] = {}
    for seed in seeds:
        t0 = time.time()
        Y = umap_layout(X, seed)
        layers = cluster_layers(Y, X)
        run = compare_run(published, layers)
        seed_runs.append(run)
        metrics["seeds"][str(seed)] = layout_metrics(X, Y, neigh_hd, classes)
        saved[f"seed_{seed}_coords"] = Y
        for j, lab in enumerate(layers):
            saved[f"seed_{seed}_layer_{j}"] = lab
        print(
            f"seed {seed}: {time.time() - t0:.0f}s, layers {run['rerun_counts']} matched {run['matched_layers']}, "
            f"AMI {[round(L['agreement']['ami'], 3) for L in run['layers']]}, "
            f"KNN k15 {metrics['seeds'][str(seed)]['knn_k15']:.3f}"
        )

    # ── subsamples ──
    sub_runs = []
    present_count = np.zeros(n, dtype=np.int32)
    for s in sub_seeds:
        t0 = time.time()
        rng = np.random.default_rng(s)
        idx = np.sort(rng.choice(n, int(round(SUBSAMPLE_FRACTION * n)), replace=False))
        present = np.zeros(n, dtype=bool)
        present[idx] = True
        present_count += present
        Y = umap_layout(X[idx], config.UMAP_RANDOM_STATE)
        layers = [expand(lab, idx, n, np.int32(-1)) for lab in cluster_layers(Y, X[idx])]
        run = compare_run(published, layers, present)
        sub_runs.append(run)
        saved[f"subsample_{s}_index"] = idx.astype(np.int32)
        saved[f"subsample_{s}_coords"] = Y
        for j, lab in enumerate(layers):
            saved[f"subsample_{s}_layer_{j}"] = lab[idx]
        print(
            f"subsample {s}: {time.time() - t0:.0f}s, {len(idx)} courses, layers {run['rerun_counts']} matched "
            f"{run['matched_layers']}, AMI {[round(L['agreement']['ami'], 3) for L in run['layers']]}"
        )

    # ── aggregate and write ──
    seed_layers, seed_regions, seed_stay = summarise(seed_runs, published, n)
    sub_layers, sub_regions, sub_stay = summarise(sub_runs, published, n)

    per_course = pd.DataFrame({"course_id": ids})
    for i in range(len(published)):
        per_course[f"stay_seeds_layer_{i}"] = seed_stay[i].astype(np.float32)
        per_course[f"stay_subsamples_layer_{i}"] = sub_stay[i].astype(np.float32)
    per_course["n_subsample_runs_present"] = present_count
    write_parquet_safely(per_course, paths["base"] / "stability.parquet")

    regions = pd.DataFrame(seed_regions).merge(
        pd.DataFrame(sub_regions), on=["layer", "cluster", "n"], suffixes=("_seeds", "_subsamples")
    )
    regions["name"] = [names[i][published[i] == r][0] for i, r in zip(regions["layer"], regions["cluster"])]
    regions = regions[
        [
            "layer",
            "cluster",
            "name",
            "n",
            "coassign_mean_seeds",
            "coassign_min_seeds",
            "coassign_mean_subsamples",
            "coassign_min_subsamples",
            "n_runs_subsamples",
        ]
    ]
    regions.to_csv(paths["base"] / "stability_regions.csv", index=False)

    metrics_meta = {
        "definitions": {
            "knn": "share of each course's k nearest neighbours (cosine, embedding space) also among its k nearest "
            "in the layout (euclidean); Kobak & Berens 2019",
            "knc": "the same for class centroids: departments (all present; and those with at least "
            f"{KNC_MIN_DEPARTMENT_SIZE} courses) and schools",
            "cpd": f"Spearman correlation of pairwise distances, {CPD_SAMPLE} sampled courses, {CPD_DRAWS} draws",
            "trustworthiness": "Venna & Kaski, as in compare_embeddings.py",
        },
        "layout_params": {
            "n_neighbors": config.UMAP_N_NEIGHBORS,
            "min_dist": config.UMAP_MIN_DIST,
            "metric": config.UMAP_METRIC,
            "published_seed": config.UMAP_RANDOM_STATE,
            "larger_n_neighbors": None if args.quick else LARGER_N_NEIGHBORS,
        },
        "seed_summary": {
            key: {
                "mean": float(
                    np.mean([m[key] if key.startswith("knn") else m[key]["mean"] for m in metrics["seeds"].values()])
                ),
                "min": float(
                    np.min([m[key] if key.startswith("knn") else m[key]["min"] for m in metrics["seeds"].values()])
                ),
                "max": float(
                    np.max([m[key] if key.startswith("knn") else m[key]["max"] for m in metrics["seeds"].values()])
                ),
            }
            for key in ("knn_k10", "knn_k15", "cpd")
        }
        | {
            key: {
                "mean": float(np.mean([m[key]["value"] for m in metrics["seeds"].values()])),
                "min": float(np.min([m[key]["value"] for m in metrics["seeds"].values()])),
                "max": float(np.max([m[key]["value"] for m in metrics["seeds"].values()])),
            }
            for key in ("knc_department", f"knc_department_min{KNC_MIN_DEPARTMENT_SIZE}", "knc_school")
        },
    }
    atomic_write_text(paths["base"] / "layout_metrics.json", json.dumps(metrics | metrics_meta, indent=1))

    import fast_hdbscan
    import sklearn
    import toponymy

    meta = {
        "reference": "published labels.parquet (labels_meta.json); layer 0 is the finest",
        "clusterer": {
            "min_clusters": config.TOPONYMY_MIN_CLUSTERS,
            "base_min_cluster_size": config.TOPONYMY_BASE_MIN_CLUSTER_SIZE,
            "deterministic_on_this_machine": bool(deterministic),
        },
        "control": {
            "what": "the published layout re-clustered on this machine; disagreement here is machine-to-machine "
            "variation in the clusterer, not a perturbation",
            "rerun_counts": control["rerun_counts"],
            "matched_layers": control["matched_layers"],
            "layers": [L["agreement"] for L in control["layers"]],
        },
        "seeds": {
            "seeds": seeds,
            "per_run": [{"matched_layers": r["matched_layers"], "rerun_counts": r["rerun_counts"]} for r in seed_runs],
            "layers": seed_layers,
        },
        "subsamples": {
            "fraction": SUBSAMPLE_FRACTION,
            "rng_seeds": sub_seeds,
            "umap_seed": config.UMAP_RANDOM_STATE,
            "per_run": [{"matched_layers": r["matched_layers"], "rerun_counts": r["rerun_counts"]} for r in sub_runs],
            "layers": sub_layers,
        },
        "layer_matching": "each published layer to the rerun layer with the nearest region count",
        "coassignment": "share of a published region's member pairs that share a rerun region; unclustered members "
        "are co-assigned with nobody; subsamples count present pairs only",
        "ami": "adjusted mutual information on courses both partitions cluster; published_members_unclustered is "
        "the share of the published layer's clustered courses the rerun leaves unclustered",
        "versions": {
            "umap_learn": umap.__version__,
            "toponymy": toponymy.__version__,
            "fast_hdbscan": getattr(fast_hdbscan, "__version__", "unknown"),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "quick": args.quick,
        "minutes": round((time.time() - t_start) / 60, 1),
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(paths["base"] / "stability_meta.json", json.dumps(meta, indent=1))

    out_npz = paths["base"] / "stability_layouts.npz"
    tmp = out_npz.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **saved)
    assert set(np.load(tmp, allow_pickle=True).files) == set(saved)
    tmp.rename(out_npz)

    print("\nper layer (published -> matched rerun counts, AMI mean [min, max], published members unclustered mean):")
    for i, (sl, bl) in enumerate(zip(seed_layers, sub_layers)):
        print(
            f"  layer {i} ({sl['published_n_clusters']} regions): seeds AMI {sl['ami']['mean']:.3f} "
            f"[{sl['ami']['min']:.3f}, {sl['ami']['max']:.3f}], unclustered {sl['published_members_unclustered']['mean']:.3f}; "
            f"subsamples AMI {bl['ami']['mean']:.3f} [{bl['ami']['min']:.3f}, {bl['ami']['max']:.3f}], "
            f"unclustered {bl['published_members_unclustered']['mean']:.3f}"
        )
    print("region co-assignment, median over regions per layer (seeds / subsamples):")
    for i, g in regions.groupby("layer"):
        print(f"  layer {i}: {g['coassign_mean_seeds'].median():.3f} / {g['coassign_mean_subsamples'].median():.3f}")
    print(f"wrote stability.parquet, stability_regions.csv, stability_meta.json, layout_metrics.json, {out_npz.name}")
    print(f"total {meta['minutes']} min")


if __name__ == "__main__":
    main()
