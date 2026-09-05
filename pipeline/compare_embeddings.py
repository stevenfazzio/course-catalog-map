"""Preregistered embedding-model comparison. Implements docs/preregistration.md exactly; runs once.

    uv run python pipeline/compare_embeddings.py --source stanford --candidates qwen3-4b arctic-l-v2 nomic-v2-moe

Writes docs/embedding_comparison.md and data/<source>/embedding_comparison.json. The metric functions are
unit-tested on synthetic data (tests/test_compare_embeddings.py); nothing here is tuned to real results.
"""

import argparse
import datetime as dt
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import config
from io_utils import atomic_write_text

K_PRIMARY = 15
K_ROBUST = 50
UMAP_SEEDS = [42, 43, 44]  # 42 is the pipeline's layout
N_BOOTSTRAP = 1_000
N_NULL = 100
RNG_SEED = 0


# ── metric ───────────────────────────────────────────────────────────────────


def knn_indices(X: np.ndarray, k: int, metric: str = "cosine") -> np.ndarray:
    """k nearest neighbours of every row, excluding the row itself. Shape (n, k)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute").fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    # The point itself is normally column 0, but exact duplicates can displace it; drop it wherever it sits.
    rows = np.arange(len(X))[:, None]
    keep = idx != rows
    out = np.empty((len(X), k), dtype=idx.dtype)
    for i in range(len(X)):
        out[i] = idx[i][keep[i]][:k]
    return out


def label_agreement(neigh: np.ndarray, labels: list[frozenset]) -> np.ndarray:
    """Per-course share of neighbours sharing at least one tag. `neigh` indexes into `labels`."""
    n, k = neigh.shape
    agree = np.zeros(n, dtype=np.float64)
    for i in range(n):
        li = labels[i]
        agree[i] = sum(1 for j in neigh[i] if li & labels[j]) / k
    return agree


def agreement_for_label(X: np.ndarray, labels: list[frozenset], k: int, metric: str = "cosine") -> np.ndarray:
    """Restrict to labelled rows (so 'no label' is never counted as disagreement), then kNN agreement."""
    mask = np.array([len(s) > 0 for s in labels])
    Xl = X[mask]
    ll = [labels[i] for i in np.flatnonzero(mask)]
    if len(Xl) <= k:
        raise ValueError(f"only {len(Xl)} labelled rows for k={k}")
    return label_agreement(knn_indices(Xl, k, metric), ll)


def trustworthiness(X: np.ndarray, Y: np.ndarray, k: int, chunk: int = 512) -> float:
    """Venna & Kaski trustworthiness of layout Y with respect to space X, identical to
    sklearn.manifold.trustworthiness(X, Y, n_neighbors=k, metric="cosine") but computed in row chunks.
    sklearn materialises two n x n distance matrices and an n x n argsort; at n = 11,500 that is several GB."""
    n = len(X)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    ind_Y = knn_indices(Y, k, metric="euclidean")
    total = 0.0
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        dist = 1.0 - Xn[rows] @ Xn.T  # cosine distance, chunk x n
        dist[np.arange(len(rows)), rows] = np.inf  # self last, as sklearn does
        order = np.argsort(dist, axis=1, kind="stable")
        inv = np.empty_like(order)
        inv[np.arange(len(rows))[:, None], order] = np.arange(1, n + 1)  # 1-based rank of every point
        ranks = inv[np.arange(len(rows))[:, None], ind_Y[rows]] - k
        total += float(ranks[ranks > 0].sum())
    return 1.0 - total * (2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> dict:
    """95% percentile interval of mean(a - b) over resampled courses; a and b are aligned per-course arrays."""
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    diff = a - b
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def null_agreement(neigh: np.ndarray, labels: list[frozenset], n: int = N_NULL, seed: int = RNG_SEED) -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        perm = rng.permutation(len(labels))
        vals.append(label_agreement(neigh, [labels[p] for p in perm]).mean())
    return float(np.mean(vals))


def neighbourhood_overlap(neigh_a: np.ndarray, neigh_b: np.ndarray) -> float:
    """Mean Jaccard of the two models' k-NN sets per course."""
    inter = np.array([len(set(a) & set(b)) for a, b in zip(neigh_a, neigh_b)])
    k = neigh_a.shape[1]
    return float((inter / (2 * k - inter)).mean())


# ── labels from the catalog ──────────────────────────────────────────────────


def label_sets(courses: pd.DataFrame) -> dict[str, list[frozenset]]:
    def tags(prefix):
        return [frozenset(g for g in gers if g.startswith(prefix)) for gers in courses["gers"]]

    return {
        "breadth (GER:DB)": tags("GER:DB-"),
        "department": [frozenset([d]) for d in courses["department_code"]],
        "school": [frozenset([s]) for s in courses["school"]],
        "WAYS": tags("WAY-"),
    }


# ── driver ───────────────────────────────────────────────────────────────────


def load_embeddings(paths: dict, key: str, course_ids: np.ndarray, raw: bool = False) -> np.ndarray:
    files = config.keyed_files(paths, key, raw=raw)
    data = np.load(files["npz"], allow_pickle=True)
    pos = pd.Series(np.arange(len(data["course_id"])), index=data["course_id"])
    missing = set(course_ids) - set(pos.index)
    assert not missing, f"{key}: {len(missing)} courses have no embedding"
    return data["embeddings"][pos.loc[course_ids].to_numpy()]


def umap_layout(X: np.ndarray, seed: int) -> np.ndarray:
    import umap

    return umap.UMAP(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        n_components=2,
        random_state=seed,
    ).fit_transform(X)


def run(courses: pd.DataFrame, embeddings: dict[str, np.ndarray], incumbent: str, layouts: bool = True) -> dict:
    labels = label_sets(courses)
    primary = "breadth (GER:DB)"
    keys = list(embeddings)
    res = {
        "n_courses": int(len(courses)),
        "n_labelled": {n: int(sum(len(s) > 0 for s in ls)) for n, ls in labels.items()},
    }

    # High-d agreement per label per k, keeping per-course arrays for the paired bootstrap.
    per_course = {}
    res["embedding_space"] = {}
    for key in keys:
        res["embedding_space"][key] = {}
        for name, ls in labels.items():
            for k in (K_PRIMARY, K_ROBUST):
                a = agreement_for_label(embeddings[key], ls, k)
                per_course[(key, name, k)] = a
                res["embedding_space"][key][f"{name} k={k}"] = float(a.mean())
    for name, ls in labels.items():
        mask = np.array([len(s) > 0 for s in ls])
        ll = [ls[i] for i in np.flatnonzero(mask)]
        neigh = knn_indices(embeddings[incumbent][mask], K_PRIMARY)
        res.setdefault("null", {})[name] = null_agreement(neigh, ll)

    res["paired_vs_incumbent"] = {
        key: paired_bootstrap(per_course[(key, primary, K_PRIMARY)], per_course[(incumbent, primary, K_PRIMARY)])
        for key in keys
        if key != incumbent
    }

    # Neighbourhood overlap between models in the embedding space.
    neigh_hd = {key: knn_indices(embeddings[key], K_PRIMARY) for key in keys}
    res["overlap_embedding_space"] = {
        f"{a} vs {b}": neighbourhood_overlap(neigh_hd[a], neigh_hd[b]) for a, b in itertools.combinations(keys, 2)
    }

    if layouts:
        res["layout_2d"] = {}
        res["trustworthiness"] = {}
        coords42 = {}
        for key in keys:
            vals = {name: [] for name in labels}
            tw = []
            for seed in UMAP_SEEDS:
                Y = umap_layout(embeddings[key], seed)
                if seed == 42:
                    coords42[key] = Y
                for name, ls in labels.items():
                    vals[name].append(float(agreement_for_label(Y, ls, K_PRIMARY, metric="euclidean").mean()))
                tw.append(trustworthiness(embeddings[key], Y, K_PRIMARY))
            res["layout_2d"][key] = {
                name: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                for name, v in vals.items()
            }
            res["trustworthiness"][key] = {
                "mean": float(np.mean(tw)),
                "min": float(np.min(tw)),
                "max": float(np.max(tw)),
            }
        neigh_2d = {key: knn_indices(coords42[key], K_PRIMARY, metric="euclidean") for key in keys}
        res["overlap_layout_2d"] = {
            f"{a} vs {b}": neighbourhood_overlap(neigh_2d[a], neigh_2d[b]) for a, b in itertools.combinations(keys, 2)
        }

    res["decision"] = decide(res, incumbent, keys, primary)
    return res


def decide(res: dict, incumbent: str, keys: list[str], primary: str) -> dict:
    """The preregistered rule, steps 2-5."""
    point = {key: res["embedding_space"][key][f"{primary} k={K_PRIMARY}"] for key in keys}
    beats = [
        key
        for key in keys
        if key != incumbent
        and res["paired_vs_incumbent"][key]["ci_low"] > 0
        and res["paired_vs_incumbent"][key]["diff"] > 0
    ]
    if not beats:
        return {"winner": incumbent, "reason": "no candidate beats the incumbent (step 4)"}
    best = max(beats, key=lambda k: point[k])
    reason = "highest point estimate among candidates that beat the incumbent (step 3)"
    # Step 3 tie rule: candidates inside each other's paired interval -> smaller model. Size order is the registry order
    # of parameter counts, approximated here by the documented family/size in the key name.
    for other in beats:
        if other != best:
            d = res["paired_vs_incumbent"]
            if (
                d[best]["ci_low"] <= d[other]["diff"] <= d[best]["ci_high"]
                and d[other]["ci_low"] <= d[best]["diff"] <= d[other]["ci_high"]
            ):
                reason += (
                    f"; {other} is inside {best}'s interval and vice versa, so the smaller model applies (step 3 tie)"
                )
    if "layout_2d" in res:
        inc = res["layout_2d"][incumbent][primary]
        win = res["layout_2d"][best][primary]["mean"]
        if win < inc["mean"] - (inc["max"] - inc["min"]):
            spread = inc["max"] - inc["min"]
            why = f"{best} fails the 2-d check (step 5): {win:.3f} vs incumbent {inc['mean']:.3f} ± {spread:.3f}"
            return {"winner": incumbent, "reason": why}
    return {"winner": best, "reason": reason}


def report(res: dict, incumbent: str, keys: list[str], corpus_note: str) -> str:
    primary = "breadth (GER:DB)"
    lines = [
        "# Embedding model comparison",
        "",
        f"Generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} by "
        f"`pipeline/compare_embeddings.py` per `docs/preregistration.md`. "
        f"Corpus: {corpus_note} ({res['n_courses']:,} courses).",
        "",
        "## Decision",
        "",
        f"**{res['decision']['winner']}**: {res['decision']['reason']}",
        "",
        "## Primary metric: embedding space, k = 15, breadth category "
        f"({res['n_labelled'][primary]:,} labelled courses; shuffled-label floor {res['null'][primary]:.3f})",
        "",
        "| model | agreement | diff vs incumbent | 95% paired CI |",
        "|---|---|---|---|",
    ]
    for key in keys:
        a = res["embedding_space"][key][f"{primary} k={K_PRIMARY}"]
        if key == incumbent:
            lines.append(f"| {key} (incumbent) | {a:.3f} | | |")
        else:
            d = res["paired_vs_incumbent"][key]
            lines.append(f"| {key} | {a:.3f} | {d['diff']:+.3f} | [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] |")
    lines += ["", "## All labels, embedding space (rows: model; floor = shuffled labels)", ""]
    names = list(res["n_labelled"])
    lines.append(
        "| model | " + " | ".join(f"{n} k=15" for n in names) + " | " + " | ".join(f"{n} k=50" for n in names) + " |"
    )
    lines.append("|---|" + "---|" * (2 * len(names)))
    lines.append(
        "| floor | " + " | ".join(f"{res['null'][n]:.3f}" for n in names) + " | " + " | ".join("" for _ in names) + " |"
    )
    for key in keys:
        e = res["embedding_space"][key]
        lines.append(
            f"| {key} | "
            + " | ".join(f"{e[f'{n} k={K_PRIMARY}']:.3f}" for n in names)
            + " | "
            + " | ".join(f"{e[f'{n} k={K_ROBUST}']:.3f}" for n in names)
            + " |"
        )
    if "layout_2d" in res:
        lines += [
            "",
            "## 2-d layout, k = 15, mean [min, max] over UMAP seeds 42/43/44; trustworthiness of the layout",
            "",
        ]
        lines.append("| model | " + " | ".join(names) + " | trustworthiness |")
        lines.append("|---|" + "---|" * (len(names) + 1))
        for key in keys:
            L = res["layout_2d"][key]
            t = res["trustworthiness"][key]
            lines.append(
                f"| {key} | "
                + " | ".join(f"{L[n]['mean']:.3f} [{L[n]['min']:.3f}, {L[n]['max']:.3f}]" for n in names)
                + f" | {t['mean']:.3f} [{t['min']:.3f}, {t['max']:.3f}] |"
            )
    lines += [
        "",
        "## Neighbourhood overlap between models (mean Jaccard of 15-NN sets)",
        "",
        "| pair | embedding space | 2-d (seed 42) |",
        "|---|---|---|",
    ]
    for pair, v in res["overlap_embedding_space"].items():
        v2 = res.get("overlap_layout_2d", {}).get(pair)
        lines.append(f"| {pair} | {v:.3f} | {'' if v2 is None else f'{v2:.3f}'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--incumbent", default=config.EMBED_MODEL_KEY)
    ap.add_argument("--candidates", nargs="+", required=True, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--raw", action="store_true", help="the sensitivity row: raw catalog + raw-text embeddings")
    ap.add_argument("--no-layouts", action="store_true", help="skip the 2-d rows (fast check)")
    ap.add_argument("--out", help="markdown report path (default docs/embedding_comparison[_raw].md)")
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    courses_path = paths["courses"] if args.raw else paths["corpus"]
    courses = pd.read_parquet(courses_path)
    ids = courses["course_id"].to_numpy()
    keys = [args.incumbent] + [c for c in args.candidates if c != args.incumbent]
    embeddings = {key: load_embeddings(paths, key, ids, raw=args.raw) for key in keys}
    print({k: v.shape for k, v in embeddings.items()})

    res = run(courses, embeddings, args.incumbent, layouts=not args.no_layouts)
    tag = "_raw" if args.raw else ""
    out_md = config.ROOT / (args.out or f"docs/embedding_comparison{tag}.md")
    atomic_write_text(out_md, report(res, args.incumbent, keys, courses_path.name))
    atomic_write_text(paths["base"] / f"embedding_comparison{tag}.json", json.dumps(res, indent=1))
    print(f"decision: {res['decision']}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
