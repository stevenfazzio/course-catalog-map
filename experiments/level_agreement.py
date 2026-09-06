"""Step 4 of docs/plan.md, check 1 (lit_review 4.3.1): kNN department and school agreement by course-number level.

    OMP_NUM_THREADS=1 uv run python experiments/level_agreement.py [--source stanford]

Pardos & Nam (2020) found, from Berkeley enrollment sequences, that lower-division courses are less disciplinarily
structured than upper-division ones. The content analogue: for every course, the share of its 15 nearest neighbours
(cosine in the embedding space; euclidean in the published 2-d layout) that belong to the same department, and the same
for school, grouped by the Bulletin's course-number levels (1-99, 100-199, 200-299, 300+). The finding replicates if
agreement rises with level.

Level is confounded with department (departments differ in size and in level mix), so three views: (1) raw means by
level with bootstrap intervals, for the undergraduate and graduate careers together and each alone; the Law, GSB and
Medicine careers number their courses on their own schemes and are tabulated but not tested; (2) size-matched chance,
the agreement random neighbours would give a course of the same department, by level; (3) within-department contrasts,
each level's difference from 1-99 after department fixed effects, over the UG and GR careers.

Writes data/<source>/validation_level.json; pipeline/07_record.py copies it into the record.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
from compare_embeddings import knn_indices  # noqa: E402
from validation_common import (  # noqa: E402
    COURSE_NUMBERING_SOURCE,
    K_NEIGHBOURS,
    LEVELS,
    bootstrap_mean,
    course_number,
    level_of,
    load_map,
    run_record,
    share_same,
    within_group_contrasts,
    write_json,
)

CAREER_SETS = {"UG+GR": ["UG", "GR"], "UG": ["UG"], "GR": ["GR"], "LAW": ["LAW"], "GSB": ["GSB"], "MED": ["MED"]}
TESTED = "UG+GR"
SPACES = ("embedding", "layout")
TARGETS = ("department", "school")


def size_matched_chance(labels: np.ndarray) -> np.ndarray:
    """Agreement a course would get from neighbours drawn at random from the rest of the corpus: (n_label - 1) / (n - 1)."""
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return (counts[inverse] - 1) / (len(labels) - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    args = ap.parse_args()
    t0 = time.time()
    corpus, X, Y = load_map(args.source)
    n = len(corpus)
    number = np.array([course_number(c) for c in corpus["primary_code"]])
    level = np.array([level_of(k) for k in number])
    career = corpus["career"].to_numpy()
    labels = {"department": corpus["department_code"].to_numpy(), "school": corpus["school"].to_numpy()}
    neigh = {
        "embedding": knn_indices(X, K_NEIGHBOURS, metric="cosine"),
        "layout": knn_indices(Y, K_NEIGHBOURS, metric="euclidean"),
    }
    agree = {(s, t): share_same(neigh[s], labels[t]) for s in SPACES for t in TARGETS}
    chance = {t: size_matched_chance(labels[t]) for t in TARGETS}
    print(
        f"{n} courses, neighbours in {time.time() - t0:.0f}s; levels {dict(zip(*np.unique(level, return_counts=True)))}"
    )

    by_level = []
    for cname, careers in CAREER_SETS.items():
        in_career = np.isin(career, careers)
        for lv in LEVELS:
            rows = in_career & (level == lv)
            if rows.sum() == 0:
                continue
            rec = {"careers": cname, "level": lv, "n": int(rows.sum())}
            for (s, t), a in agree.items():
                rec[f"{s}_{t}"] = bootstrap_mean(a[rows])
            for t in TARGETS:
                rec[f"chance_{t}"] = float(chance[t][rows].mean())
            by_level.append(rec)
    for rec in by_level:
        if rec["careers"] in ("UG+GR", "UG", "GR"):
            print(
                f"  {rec['careers']:5} {rec['level']:8} n={rec['n']:5d}  "
                + "  ".join(f"{s[:3]}/{t[:4]} {rec[f'{s}_{t}']['mean']:.3f}" for s in SPACES for t in TARGETS)
                + f"  chance dept {rec['chance_department']:.3f} school {rec['chance_school']:.3f}"
            )

    tested = np.isin(career, CAREER_SETS[TESTED])
    present = [lv for lv in LEVELS if (tested & (level == lv)).sum() > 0]
    contrasts = {}
    for (s, t), a in agree.items():
        contrasts[f"{s}_{t}"] = within_group_contrasts(a[tested], labels["department"][tested], level[tested], present)
        print(
            f"  within-department {s}/{t}: "
            + ", ".join(
                f"{lv} {c['estimate']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
                for lv, c in contrasts[f"{s}_{t}"]["contrasts"].items()
            )
        )

    by_career = []
    for c in ("UG", "GR", "LAW", "GSB", "MED"):
        rows = career == c
        rec = {"career": c, "n": int(rows.sum())}
        for (s, t), a in agree.items():
            rec[f"{s}_{t}"] = bootstrap_mean(a[rows])
        by_career.append(rec)

    out = run_record(
        args.source,
        k=K_NEIGHBOURS,
        levels=LEVELS,
        course_numbering_source=COURSE_NUMBERING_SOURCE,
        tested_careers=CAREER_SETS[TESTED],
        definitions={
            "agreement": f"share of a course's {K_NEIGHBOURS} nearest neighbours with the same department (or school); "
            "embedding space is cosine over the unit-norm Qwen3-0.6B embeddings, layout is euclidean in the published "
            "2-d coordinates; neighbours are drawn from the whole corpus regardless of level",
            "chance": "size-matched chance: (n_same_label - 1) / (n - 1) per course, averaged over the group",
            "within_department_contrasts": "OLS of per-course agreement on level dummies with department fixed effects "
            "(within-department demeaning), baseline 1-99, over the UG and GR careers; 95% percentile bootstrap over "
            "courses (1,000 resamples)",
            "level": "from the number in primary_code; the Bulletin's own breakpoints",
        },
        n_courses=n,
        level_counts={lv: int((level == lv).sum()) for lv in LEVELS},
        by_level=by_level,
        within_department_contrasts=contrasts,
        by_career=by_career,
        seconds=round(time.time() - t0),
    )
    write_json(config.source_paths(args.source)["base"] / "validation_level.json", out)
    print(f"wrote validation_level.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
