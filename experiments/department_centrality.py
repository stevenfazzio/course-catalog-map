"""Step 4 of docs/plan.md, check 4 (lit_review 4.3.3): which departments are central in the department-centroid graph?

    OMP_NUM_THREADS=1 uv run python experiments/department_centrality.py [--source stanford]

Gim et al. (2025), on Korean course names, found engineering departments at the core of the department similarity
network and the natural sciences at its periphery. Here the network is the department-centroid cosine similarity matrix
the structure report already builds (departments with at least MIN_COURSES courses, centroid = unit-norm mean embedding
over the department's courses by primary code). Four centralities: strength (mean similarity to every other
department), eigenvector centrality of the weighted similarity matrix, in-degree in the directed k-nearest-department
graph (how many departments count this one among their k nearest) and closeness in its undirected version. The same
in-degree and closeness are computed from centroid distances in the published 2-d layout. A bootstrap over courses
within department gives each department's strength-rank interval.

Summaries by school and, because the claim names the natural sciences, by the three divisions of Humanities & Sciences
(Humanities and Arts, Natural Sciences, Social Sciences) for the school's departments; its interdisciplinary programmes
and the Language Center are grouped as "H&S programs and centers". The division membership is the department list on
Wikipedia's page for the school (read 2026-09-06; the school's own site names the divisions but did not list members on
the pages fetched), applied to the catalog's academicOrganization codes and recorded in the output.

Writes data/<source>/validation_centrality.json and validation_centrality.csv (per department); 07_record.py copies them.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
from validation_common import RNG_SEED, bootstrap_mean, load_map, run_record, write_json  # noqa: E402

MIN_COURSES = 15  # the structure report's threshold
K_NEAREST = 5
N_BOOT = 200
HS_DIVISION_SOURCE = (
    "https://en.wikipedia.org/wiki/Stanford_University_School_of_Humanities_and_Sciences (read 2026-09-06)"
)
# academicOrganization code -> division, for the departments Wikipedia lists under each division.
HS_DIVISIONS = {
    **dict.fromkeys(
        [
            "ART",
            "CLASSICS",
            "DRAMA",
            "ASIANLANG",
            "ENGLISH",
            "HISTORY",
            "LINGUISTIC",
            "MUSIC",
            "PHILOSOPHY",
            "RELIGST",
            "COMPARLIT",
            "FRENCHITAL",
            "GERMANSTU",
            "SPANPORT",
            "SLAVIC",
        ],
        "H&S Humanities and Arts",
    ),
    **dict.fromkeys(["APPLPHYSIC", "BIO", "CHEMISTRY", "MATH", "PHYSICS", "STATISTICS"], "H&S Natural Sciences"),
    **dict.fromkeys(
        ["ANTHRO", "COMMUN", "ECONOMICS", "POLITSCI", "PSYCHOLOGY", "STS", "SOCIOLOGY"], "H&S Social Sciences"
    ),
}
HS_OTHER = "H&S programs and centers"
MEASURES = ["strength", "eigenvector", "indegree", "closeness"]


def centralities(S: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Strength, eigenvector centrality, k-nearest in-degree and closeness from a similarity matrix (diagonal ignored)."""
    n = len(S)
    off = ~np.eye(n, dtype=bool)
    A = np.where(off, np.clip(S, 0, None), 0.0)
    strength = A.sum(axis=1) / (n - 1)
    w, v = np.linalg.eigh(A)
    vec = np.abs(v[:, np.argmax(w)])
    eigen = vec / vec.sum()
    masked = np.where(off, S, -np.inf)
    nearest = np.argsort(-masked, axis=1)[:, :k]
    indegree = np.bincount(nearest.ravel(), minlength=n).astype(np.float64)
    adj = np.zeros((n, n))
    adj[np.repeat(np.arange(n), k), nearest.ravel()] = 1.0
    adj = np.maximum(adj, adj.T)
    dist = shortest_path(adj, unweighted=True, directed=False)
    reach = np.isfinite(dist).sum(axis=1) - 1
    sumd = np.where(np.isfinite(dist), dist, 0.0).sum(axis=1)
    closeness = np.where(sumd > 0, (reach / (n - 1)) * (reach / np.where(sumd > 0, sumd, 1.0)), 0.0)
    return {"strength": strength, "eigenvector": eigen, "indegree": indegree, "closeness": closeness}


def rank_desc(x: np.ndarray) -> np.ndarray:
    """1 = largest."""
    return (np.argsort(np.argsort(-x)) + 1).astype(int)


def group_summary(df: pd.DataFrame, column: str) -> list[dict]:
    out = []
    for g, d in df.groupby(column, sort=False):
        rec = {"group": g, "n_departments": int(len(d))}
        for m in MEASURES:
            rec[m] = bootstrap_mean(d[m].to_numpy())
            rec[f"mean_rank_{m}"] = float(d[f"rank_{m}"].mean())
        out.append(rec)
    return sorted(out, key=lambda r: r["mean_rank_strength"])


def versus_rest(df: pd.DataFrame, column: str, group: str, measure: str = "strength") -> dict:
    a, b = df.loc[df[column] == group, measure].to_numpy(), df.loc[df[column] != group, measure].to_numpy()
    if len(a) == 0 or len(b) == 0:
        return {"group": group, "n": int(len(a)), "prob_higher": None, "p": None}
    u = mannwhitneyu(a, b)
    return {
        "group": group,
        "n": int(len(a)),
        "prob_higher": float(u.statistic / (len(a) * len(b))),
        "p": float(u.pvalue),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    args = ap.parse_args()
    t0 = time.time()
    corpus, X, Y = load_map(args.source)
    dept = corpus["department_code"].to_numpy()
    counts = corpus["department_code"].value_counts()
    codes = sorted(counts[counts >= MIN_COURSES].index)
    members = [np.flatnonzero(dept == c) for c in codes]
    info = corpus.groupby("department_code").agg(
        department=("department", "first"),
        school=("school", "first"),
        academic_group=("academic_group", "first"),
        academic_organization=("academic_organization", lambda s: s.mode().iloc[0]),
    )
    print(
        f"{len(codes)} departments with >= {MIN_COURSES} courses ({int(counts[codes].sum())} of {len(corpus)} courses)"
    )

    def centroids(idx_lists, V, normalise=True):
        C = np.stack([V[m].mean(axis=0) for m in idx_lists])
        return C / np.linalg.norm(C, axis=1, keepdims=True) if normalise else C

    C = centroids(members, X)
    S = C @ C.T
    emb = centralities(S, K_NEAREST)
    Cy = centroids(members, Y, normalise=False)
    D = np.linalg.norm(Cy[:, None, :] - Cy[None, :, :], axis=2)
    lay = centralities(-D, K_NEAREST)  # similarity = negative distance; strength/eigenvector are meaningless here

    rng = np.random.default_rng(RNG_SEED)
    boot_ranks = np.empty((N_BOOT, len(codes)))
    for b in range(N_BOOT):
        resampled = [m[rng.integers(0, len(m), len(m))] for m in members]
        Cb = centroids(resampled, X)
        boot_ranks[b] = rank_desc(centralities(Cb @ Cb.T, K_NEAREST)["strength"])

    df = pd.DataFrame({"department_code": codes})
    df["department"] = [info.loc[c, "department"] for c in codes]
    df["school"] = [info.loc[c, "school"] for c in codes]
    df["academic_organization"] = [info.loc[c, "academic_organization"] for c in codes]
    df["n_courses"] = [len(m) for m in members]
    df["spread"] = [
        float(np.mean(1.0 - X[m] @ C[i])) for i, m in enumerate(members)
    ]  # mean cosine distance of members to their centroid
    df["division"] = [
        HS_DIVISIONS.get(info.loc[c, "academic_organization"], HS_OTHER)
        if info.loc[c, "academic_group"] == "H&S"
        else info.loc[c, "school"]
        for c in codes
    ]
    for m in MEASURES:
        df[m] = emb[m]
        df[f"rank_{m}"] = rank_desc(emb[m])
    df["rank_strength_ci_low"] = np.percentile(boot_ranks, 2.5, axis=0).astype(int)
    df["rank_strength_ci_high"] = np.percentile(boot_ranks, 97.5, axis=0).astype(int)
    df["nearest_department"] = [
        codes[j] for j in np.argsort(-np.where(np.eye(len(codes), dtype=bool), -np.inf, S), axis=1)[:, 0]
    ]
    df["layout_indegree"] = lay["indegree"]
    df["layout_closeness"] = lay["closeness"]
    df = df.sort_values("rank_strength")
    base = config.source_paths(args.source)["base"]
    df.to_csv(base / "validation_centrality.csv", index=False)

    by_school = group_summary(df, "school")
    by_division = group_summary(df, "division")
    comparisons = {
        "school": [versus_rest(df, "school", g) for g in df["school"].unique()],
        "division": [versus_rest(df, "division", g) for g in df["division"].unique()],
    }
    print("  by school (mean strength rank, mean strength):")
    for r in by_school:
        print(
            f"    {r['group'][:45]:45} n={r['n_departments']:3d} rank {r['mean_rank_strength']:6.1f} strength {r['strength']['mean']:.3f} [{r['strength']['ci_low']:.3f}, {r['strength']['ci_high']:.3f}] indeg {r['indegree']['mean']:.1f}"
        )
    print("  by division:")
    for r in by_division:
        print(
            f"    {r['group'][:45]:45} n={r['n_departments']:3d} rank {r['mean_rank_strength']:6.1f} strength {r['strength']['mean']:.3f} [{r['strength']['ci_low']:.3f}, {r['strength']['ci_high']:.3f}] indeg {r['indegree']['mean']:.1f}"
        )
    print(
        "  top 10 by strength: "
        + ", ".join(
            f"{c} ({s:.3f}, rank CI {lo}-{hi})"
            for c, s, lo, hi in df[["department_code", "strength", "rank_strength_ci_low", "rank_strength_ci_high"]]
            .head(10)
            .itertuples(index=False)
        )
    )
    print(
        "  bottom 10 by strength: "
        + ", ".join(f"{c} ({s:.3f})" for c, s in df[["department_code", "strength"]].tail(10).itertuples(index=False))
    )
    rank_corr = df[[f"rank_{m}" for m in MEASURES]].corr(method="spearman")
    spread_rho = {m: float(df[[m, "spread"]].corr(method="spearman").iloc[0, 1]) for m in MEASURES}
    print(
        f"  Spearman of each measure with within-department spread: {', '.join(f'{m} {r:+.2f}' for m, r in spread_rho.items())}"
    )
    print(f"  Spearman between measure ranks:\n{rank_corr.round(2).to_string()}")

    out = run_record(
        args.source,
        min_courses=MIN_COURSES,
        k_nearest=K_NEAREST,
        n_bootstrap=N_BOOT,
        hs_division_source=HS_DIVISION_SOURCE,
        hs_divisions=HS_DIVISIONS,
        hs_other=HS_OTHER,
        definitions={
            "centroid": "unit-norm mean embedding over the department's courses by primary code",
            "strength": "mean cosine similarity to every other department centroid",
            "eigenvector": "principal eigenvector of the similarity matrix with zero diagonal, scaled to sum 1",
            "indegree": f"number of departments whose {K_NEAREST} most similar departments include this one",
            "closeness": f"Wasserman-Faust closeness in the undirected {K_NEAREST}-nearest-department graph",
            "layout_*": "the same in-degree and closeness from euclidean centroid distances in the published 2-d layout",
            "spread": "mean cosine distance of a department's courses to its own centroid: a heterogeneous department's centroid sits near the global mean and is similar to everyone",
            "rank_strength_ci": f"95% interval of the strength rank over {N_BOOT} bootstraps of courses within department",
            "prob_higher": "P(a department in the group has higher strength than one outside it), Mann-Whitney U / (n1 n0)",
        },
        n_departments=len(codes),
        n_courses_covered=int(counts[codes].sum()),
        by_school=by_school,
        by_division=by_division,
        versus_rest=comparisons,
        measure_rank_spearman=rank_corr.round(4).to_dict(),
        spread_spearman=spread_rho,
        top_10_strength=df.head(10)[
            ["department_code", "department", "school", "strength", "rank_strength_ci_low", "rank_strength_ci_high"]
        ].to_dict("records"),
        bottom_10_strength=df.tail(10)[
            ["department_code", "department", "school", "strength", "rank_strength_ci_low", "rank_strength_ci_high"]
        ].to_dict("records"),
        seconds=round(time.time() - t0),
    )
    write_json(base / "validation_centrality.json", out)
    print(f"wrote validation_centrality.json and validation_centrality.csv in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
