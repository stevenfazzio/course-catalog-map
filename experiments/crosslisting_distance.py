"""Step 4 of docs/plan.md, check 2 (lit_review 4.3.4 and 4.2.2): does cross-listing follow content proximity?

    OMP_NUM_THREADS=1 uv run python experiments/crosslisting_distance.py [--source stanford]

Jacobs (2014) reads cross-listing as the traditional observable of interdisciplinarity. Two content-side questions:
(a) department pairs: do departments that share cross-listed courses have closer centroids than pairs that do not, once
school is controlled for? (b) courses: do cross-listed courses sit further from their home department's core than
single-listed ones (the displacement of 4.2.2), and does displacement grow with the number of listings?

Circularity guard: every centroid is the unit-norm mean of a department's single-listed courses only (n_listings == 1),
so no cross-listed course contributes to a centroid it is compared against, and a single-listed course is compared
against a centroid that leaves it out. Departments need at least MIN_SINGLE_LISTED such courses. The naive centroid over
every course by primary department is reported as a sensitivity row.

Pairs: mean centroid distance for linked (sharing at least one cross-listed course) and unlinked pairs, overall and
within same-school and cross-school pairs; the probability that a random linked pair is closer than a random unlinked
pair; the linked coefficient after school-pair fixed effects, with a bootstrap interval over pairs (pairs sharing a
department are not independent, so the interval is approximate) and a permutation p-value that reassigns links at
random within each school pair; and, among linked pairs, the rank correlation of shared-course count with distance.

Writes data/<source>/validation_crosslisting.json, validation_crosslisting_pairs.csv (one row per department pair) and
crosslisting_displacement.parquet (one row per course, aligned to corpus.parquet); pipeline/07_record.py copies them.
"""

import argparse
import collections
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
from io_utils import write_parquet_safely  # noqa: E402
from validation_common import (  # noqa: E402
    RNG_SEED,
    bootstrap_mean,
    fixed_effects_fit,
    load_map,
    run_record,
    within_group_contrasts,
    write_json,
)

MIN_SINGLE_LISTED = 10
LISTING_BUCKETS = ["1", "2", "3", "4+"]
SHARED_BUCKETS = [(1, 1, "1"), (2, 4, "2-4"), (5, 10_000, "5+")]
N_PERMUTATIONS = 1_000


def listing_bucket(n: int) -> str:
    return "4+" if n >= 4 else str(n)


def shared_counts(listed_in, depts: set[str]) -> dict[tuple[str, str], int]:
    """Number of courses cross-listed between each pair of departments in `depts` (pair keys sorted)."""
    counts: collections.Counter = collections.Counter()
    for li in listed_in:
        for a, b in itertools.combinations(sorted(set(li) & depts), 2):
            counts[(a, b)] += 1
    return dict(counts)


def pair_table(codes, C_held, C_naive, school_of: dict, shared: dict) -> pd.DataFrame:
    S_h, S_n = C_held @ C_held.T, C_naive @ C_naive.T
    rows = []
    for i, j in itertools.combinations(range(len(codes)), 2):
        a, b = codes[i], codes[j]
        rows.append(
            {
                "dept_a": a,
                "dept_b": b,
                "school_a": school_of[a],
                "school_b": school_of[b],
                "n_shared": shared.get((a, b), 0),
                "distance_heldout": 1.0 - S_h[i, j],
                "distance_naive": 1.0 - S_n[i, j],
            }
        )
    df = pd.DataFrame(rows)
    df["same_school"] = df["school_a"] == df["school_b"]
    df["linked"] = df["n_shared"] > 0
    df["school_pair"] = ["|".join(sorted((x, y))) for x, y in zip(df["school_a"], df["school_b"])]
    return df


def prob_closer(linked: np.ndarray, unlinked: np.ndarray) -> float:
    """P(a random linked pair's distance < a random unlinked pair's), ties counted half (Mann-Whitney U / n1 n0)."""
    return float(mannwhitneyu(unlinked, linked).statistic / (len(linked) * len(unlinked)))


def permute_within(rng, x: np.ndarray, strata: np.ndarray) -> np.ndarray:
    out = x.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = x[idx][rng.permutation(len(idx))]
    return out


def linked_effect(df: pd.DataFrame, column: str, n_perm: int = N_PERMUTATIONS, seed: int = RNG_SEED) -> dict:
    """The linked coefficient on `column` with school-pair fixed effects: bootstrap interval over pairs and a one-sided
    permutation p-value (links reassigned at random within each school pair; linked pairs closer means negative)."""
    y = df[column].to_numpy(dtype=np.float64)
    strata = df["school_pair"].to_numpy()
    level = np.where(df["linked"].to_numpy(), "linked", "unlinked")
    fe = within_group_contrasts(y, strata, level, ["unlinked", "linked"])
    obs = fe["contrasts"]["linked"]["estimate"]
    rng = np.random.default_rng(seed)
    linked = df["linked"].to_numpy()
    perms = np.array(
        [
            fixed_effects_fit(y, strata, permute_within(rng, linked, strata)[:, None].astype(np.float64))[0]
            for _ in range(n_perm)
        ]
    )
    return {
        "estimate": obs,
        "ci_low": fe["contrasts"]["linked"]["ci_low"],
        "ci_high": fe["contrasts"]["linked"]["ci_high"],
        "n_school_pairs": fe["n_groups"],
        "permutation_p": float((np.sum(perms <= obs) + 1) / (n_perm + 1)),
        "permutation_null_mean": float(perms.mean()),
        "permutation_null_sd": float(perms.std()),
    }


def stratum_summary(df: pd.DataFrame, column: str) -> dict:
    out = {}
    for name, rows in (
        ("all", np.ones(len(df), bool)),
        ("same_school", df["same_school"].to_numpy()),
        ("cross_school", ~df["same_school"].to_numpy()),
    ):
        d = df[rows]
        li, un = d.loc[d["linked"], column].to_numpy(), d.loc[~d["linked"], column].to_numpy()
        out[name] = {
            "n_pairs": int(len(d)),
            "n_linked": int(len(li)),
            "linked": bootstrap_mean(li),
            "unlinked": bootstrap_mean(un),
            "prob_linked_closer": prob_closer(li, un) if len(li) and len(un) else None,
        }
    return out


def displacement_to(
    X: np.ndarray, sums: np.ndarray, counts: np.ndarray, dept_index: np.ndarray, leave_out: np.ndarray
) -> np.ndarray:
    """Cosine distance from each course to its department centroid; the course's own vector is removed from the centroid
    where `leave_out` is set (it contributed to the sum). NaN where dept_index < 0."""
    out = np.full(len(X), np.nan)
    ok = dept_index >= 0
    v = sums[dept_index[ok]].copy()
    c = counts[dept_index[ok]].astype(np.float64)
    lo = leave_out[ok]
    v[lo] -= X[ok][lo]
    c[lo] -= 1
    v /= c[:, None]
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out[ok] = 1.0 - np.einsum("ij,ij->i", X[ok], v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    args = ap.parse_args()
    t0 = time.time()
    corpus, X, _ = load_map(args.source)
    n = len(corpus)
    dept = corpus["department_code"].to_numpy()
    n_listings = corpus["n_listings"].to_numpy()
    single = n_listings == 1
    listed_in = [list(li) for li in corpus["listed_in"]]
    school_of = corpus.groupby("department_code")["school"].first().to_dict()

    # ── centroids: held-out (single-listed courses only) and naive (every course by primary department) ──
    codes_all, inv_all, n_all = np.unique(dept, return_inverse=True, return_counts=True)
    sums_all = np.zeros((len(codes_all), X.shape[1]))
    np.add.at(sums_all, inv_all, X)
    sums_single = np.zeros_like(sums_all)
    np.add.at(sums_single, inv_all[single], X[single])
    n_single = np.bincount(inv_all[single], minlength=len(codes_all))
    keep = n_single >= MIN_SINGLE_LISTED
    codes = [str(c) for c in codes_all[keep]]
    C_held = sums_single[keep] / n_single[keep, None]
    C_held /= np.linalg.norm(C_held, axis=1, keepdims=True)
    C_naive = sums_all[keep] / n_all[keep, None]
    C_naive /= np.linalg.norm(C_naive, axis=1, keepdims=True)
    print(
        f"{n} courses; {len(codes)} of {len(codes_all)} departments have >= {MIN_SINGLE_LISTED} single-listed courses ({int(n_all[keep].sum())} courses)"
    )

    # ── (a) department pairs ──
    shared = shared_counts(listed_in, set(codes))
    pairs = pair_table(codes, C_held, C_naive, school_of, shared)
    pairs.to_csv(config.source_paths(args.source)["base"] / "validation_crosslisting_pairs.csv", index=False)
    print(
        f"{len(pairs)} pairs, {int(pairs['linked'].sum())} linked ({int((pairs['linked'] & pairs['same_school']).sum())} same-school); {int(pairs['same_school'].sum())} same-school pairs"
    )
    pair_result = {}
    for column in ("distance_heldout", "distance_naive"):
        strata = stratum_summary(pairs, column)
        effect = linked_effect(pairs, column)
        li = pairs[pairs["linked"]]
        dose = spearmanr(li["n_shared"], li[column])
        by_shared = []
        for lo, hi, name in SHARED_BUCKETS:
            rows = (pairs["n_shared"] >= lo) & (pairs["n_shared"] <= hi)
            by_shared.append({"n_shared": name, **bootstrap_mean(pairs.loc[rows, column].to_numpy())})
        by_shared.insert(0, {"n_shared": "0", **bootstrap_mean(pairs.loc[~pairs["linked"], column].to_numpy())})
        pair_result[column] = {
            "by_stratum": strata,
            "linked_effect_school_pair_fixed_effects": effect,
            "dose_spearman_among_linked": {"rho": float(dose.statistic), "p": float(dose.pvalue), "n": int(len(li))},
            "by_n_shared": by_shared,
        }
        print(
            f"  {column}: linked {strata['all']['linked']['mean']:.3f} vs unlinked {strata['all']['unlinked']['mean']:.3f} "
            f"(P closer {strata['all']['prob_linked_closer']:.3f}); same-school {strata['same_school']['linked']['mean']:.3f} vs {strata['same_school']['unlinked']['mean']:.3f} "
            f"(P {strata['same_school']['prob_linked_closer']:.3f}); cross-school {strata['cross_school']['linked']['mean']:.3f} vs {strata['cross_school']['unlinked']['mean']:.3f} "
            f"(P {strata['cross_school']['prob_linked_closer']:.3f}); FE {effect['estimate']:+.4f} [{effect['ci_low']:+.4f}, {effect['ci_high']:+.4f}] perm p {effect['permutation_p']:.4f}; "
            f"dose rho {dose.statistic:+.3f}"
        )

    # ── (b) displacement per course ──
    code_pos = {c: i for i, c in enumerate(codes)}
    dept_index = np.array([code_pos.get(d, -1) for d in dept])
    disp_held = displacement_to(X, sums_single[keep], n_single[keep], dept_index, leave_out=single)
    disp_naive = displacement_to(X, sums_all[keep], n_all[keep], dept_index, leave_out=np.ones(n, bool))
    bucket = np.array([listing_bucket(k) for k in n_listings])
    spans_schools = np.array(
        [len({school_of[d] for d in li if d in school_of}) > 1 for li in listed_in]
    )  # a few listing departments have no primary course
    nearest_listed = np.array([None] * n, dtype=object)
    for i in range(n):
        if n_listings[i] > 1 and dept_index[i] >= 0:
            cands = [d for d in listed_in[i] if d in code_pos]
            if len(cands) >= 2:
                sims = [float(X[i] @ C_held[code_pos[d]]) for d in cands]
                nearest_listed[i] = cands[int(np.argmax(sims))]
    has_nearest = np.array([v is not None for v in nearest_listed])
    home_is_nearest = pd.array(
        [None if v is None else bool(v == d) for v, d in zip(nearest_listed, dept)], dtype="boolean"
    )
    n_candidates = np.array(
        [
            len([d for d in li if d in code_pos]) if n_listings[i] > 1 and dept_index[i] >= 0 else 0
            for i, li in enumerate(listed_in)
        ]
    )

    covered = dept_index >= 0
    disp_result = {}
    for name, disp in (("heldout", disp_held), ("naive", disp_naive)):
        by_bucket = [{"n_listings": b, **bootstrap_mean(disp[covered & (bucket == b)])} for b in LISTING_BUCKETS]
        present = [b for b in LISTING_BUCKETS if (covered & (bucket == b)).sum() > 0]
        contrasts = within_group_contrasts(disp[covered], dept[covered], bucket[covered], present)
        cross = covered & (n_listings > 1)
        disp_result[name] = {
            "by_n_listings": by_bucket,
            "within_department_contrasts": contrasts,
            "cross_listed_spanning_schools": bootstrap_mean(disp[cross & spans_schools]),
            "cross_listed_within_school": bootstrap_mean(disp[cross & ~spans_schools]),
        }
        print(
            f"  displacement ({name}): "
            + ", ".join(f"{r['n_listings']} {r['mean']:.3f}" for r in by_bucket)
            + "; within-department "
            + ", ".join(
                f"{b} {c['estimate']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
                for b, c in contrasts["contrasts"].items()
            )
            + f"; spanning schools {disp_result[name]['cross_listed_spanning_schools']['mean']:.3f} vs within {disp_result[name]['cross_listed_within_school']['mean']:.3f}"
        )
    n_hn = int(has_nearest.sum())
    share_home_nearest = float(home_is_nearest[has_nearest].mean()) if n_hn else None
    chance_home_nearest = (
        float((1.0 / n_candidates[has_nearest]).mean()) if n_hn else None
    )  # if the home were a random listed department
    print(
        f"  cross-listed courses with >= 2 listed departments among the {len(codes)}: {n_hn}; home department is the nearest listed centroid for {share_home_nearest:.3f} (random home {chance_home_nearest:.3f})"
    )

    per_course = pd.DataFrame(
        {
            "course_id": corpus["course_id"].to_numpy(),
            "department_code": dept,
            "n_listings": n_listings,
            "listing_bucket": bucket,
            "displacement": disp_held.astype(np.float32),
            "displacement_naive": disp_naive.astype(np.float32),
            "spans_schools": spans_schools,
            "nearest_listed_department": nearest_listed,
            "home_is_nearest_listed": home_is_nearest,
        }
    )
    write_parquet_safely(per_course, config.source_paths(args.source)["base"] / "crosslisting_displacement.parquet")

    out = run_record(
        args.source,
        min_single_listed=MIN_SINGLE_LISTED,
        n_permutations=N_PERMUTATIONS,
        definitions={
            "centroid_heldout": "unit-norm mean embedding of a department's single-listed courses (n_listings == 1); "
            "a single-listed course's own displacement leaves it out of the centroid",
            "centroid_naive": "unit-norm mean over every course by primary department (leave-one-out for displacement)",
            "distance": "cosine distance, 1 - cos",
            "linked": "the two departments share at least one cross-listed course (both codes in listed_in)",
            "prob_linked_closer": "P(random linked pair closer than random unlinked pair), Mann-Whitney U / (n1 n0)",
            "linked_effect": "OLS of pair distance on the linked indicator with school-pair fixed effects; percentile "
            "bootstrap over pairs (approximate: pairs sharing a department are dependent); permutation p reassigns "
            "the linked indicator at random within each school pair, one-sided (linked closer)",
            "displacement": "cosine distance from a course to its home department's held-out centroid",
            "within_department_contrasts": "OLS of displacement on n_listings-bucket dummies with department fixed "
            "effects, baseline single-listed; bootstrap over courses",
            "spans_schools": "the course's listing departments belong to more than one school (departments with no primary course, hence no school here, ignored)",
            "home_is_nearest_listed": "for a cross-listed course with two or more listed departments among those with "
            "centroids, whether the primary department's centroid is the closest of them; share_if_home_were_random is "
            "the mean of 1 / (number of such departments)",
        },
        n_courses=n,
        n_departments_with_centroids=len(codes),
        n_courses_covered=int(covered.sum()),
        n_pairs=int(len(pairs)),
        n_linked_pairs=int(pairs["linked"].sum()),
        pairs=pair_result,
        displacement=disp_result,
        nearest_listed={
            "n_courses": n_hn,
            "share_home_is_nearest": share_home_nearest,
            "share_if_home_were_random": chance_home_nearest,
        },
        seconds=round(time.time() - t0),
    )
    write_json(config.source_paths(args.source)["base"] / "validation_crosslisting.json", out)
    print(
        f"wrote validation_crosslisting.json, validation_crosslisting_pairs.csv, crosslisting_displacement.parquet in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
