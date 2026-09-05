"""Exploratory phase helper (docs/preregistration.md): what is inside each named region, in terms of content
and metadata, so that any hygiene rule can be written against the data rather than against map position.

    uv run python experiments/explore_regions.py --embedding arctic-l-v2 [--layer 0] [--top 25]

Reports, per region of the chosen layer: size, dominant departments, unscheduled share, empty-description
share, the most common description prefix and its share (boilerplate detector), and title-pattern hits.
Then lists the regions that look like boilerplate rather than subject matter.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

TITLE_PATTERNS = {
    "independent study": r"\bindependent (?:study|studies|research|work)\b|\bdirected (?:reading|research|study)\b",
    "thesis/dissertation": r"\b(?:thesis|dissertation|TGR)\b",
    "research": r"\bresearch\b",
    "practicum/internship": r"\b(?:practicum|internship|clerkship|clinical rotation)\b",
    "seminar/colloquium": r"\b(?:seminar|colloquium)\b",
    "special topics": r"\b(?:special topics|topics in|selected topics|advanced topics)\b",
    "teaching": r"\b(?:teaching|pedagogy)\b",
    "language course": r"\b(?:first|second|third|fourth)[- ]year\b|\b(?:beginning|intermediate|advanced) [A-Z][a-z]+\b",
}


def describe_region(sub: pd.DataFrame) -> dict:
    prefix = sub["description"].str[:100]
    top_prefix, top_n = ("", 0) if prefix.empty else Counter(prefix).most_common(1)[0]
    hits = {
        name: int(sub["title"].str.contains(pat, case=False, regex=True).sum()) for name, pat in TITLE_PATTERNS.items()
    }
    depts = Counter(sub["department_code"]).most_common(3)
    return {
        "n": len(sub),
        "depts": ", ".join(f"{d} {n}" for d, n in depts),
        "n_depts": sub["department_code"].nunique(),
        "unscheduled": float((~sub["scheduled"]).mean()),
        "empty_desc": float((sub["description"] == "").mean()),
        "desc_words": float(sub["description"].str.split().str.len().median()),
        "top_prefix_share": top_n / max(len(sub), 1),
        "top_prefix": top_prefix[:70],
        "title_hits": {k: v for k, v in hits.items() if v / max(len(sub), 1) >= 0.5},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY)
    ap.add_argument("--layer", type=int, default=0, help="0 = finest")
    ap.add_argument("--top", type=int, default=25, help="how many suspicious regions to print")
    ap.add_argument("--raw", action="store_true", help="the raw-catalog build (the exploration maps)")
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    files = config.keyed_files(paths, args.embedding, raw=args.raw)
    courses = pd.read_parquet(paths["courses"] if args.raw else paths["corpus"])
    labels = pd.read_parquet(files["labels"])
    assert (labels["course_id"].to_numpy() == courses["course_id"].to_numpy()).all()
    col = f"label_layer_{args.layer}"
    df = courses.assign(region=labels[col].to_numpy(), cluster=labels[f"cluster_layer_{args.layer}"].to_numpy())

    print(
        f"{args.embedding}, layer {args.layer}: {df['cluster'].max() + 1} regions, {(df['cluster'] < 0).mean():.1%} unlabelled"
    )
    unl = df[df["cluster"] < 0]
    print(
        "unlabelled by school:",
        {
            k: f"{v:.0%}"
            for k, v in (unl["school"].value_counts() / df["school"].value_counts())
            .sort_values(ascending=False)
            .head(5)
            .items()
        },
    )
    print(
        f"unlabelled: unscheduled {(~unl['scheduled']).mean():.0%} vs overall {(~df['scheduled']).mean():.0%}; "
        f"empty desc {(unl['description'] == '').mean():.1%} vs overall {(df['description'] == '').mean():.1%}"
    )

    rows = []
    for (cl, name), sub in df[df["cluster"] >= 0].groupby(["cluster", "region"]):
        rows.append({"cluster": cl, "region": name} | describe_region(sub))
    reg = pd.DataFrame(rows)

    print(f"\n── regions where one description prefix covers ≥40% (boilerplate candidates), top {args.top} ──")
    for _, r in reg[reg["top_prefix_share"] >= 0.4].sort_values("n", ascending=False).head(args.top).iterrows():
        print(
            f"  [{r['cluster']:3d}] {r['region'][:60]:60} n={r['n']:3d} {r['top_prefix_share']:.0%} share | {r['depts']} | {r['top_prefix']}"
        )

    print(f"\n── regions where a title pattern covers ≥50%, top {args.top} ──")
    for _, r in reg[reg["title_hits"].map(bool)].sort_values("n", ascending=False).head(args.top).iterrows():
        print(f"  [{r['cluster']:3d}] {r['region'][:60]:60} n={r['n']:3d} {r['title_hits']} | {r['depts']}")

    print("\n── regions with ≥80% unscheduled or ≥30% empty descriptions ──")
    for _, r in (
        reg[(reg["unscheduled"] >= 0.8) | (reg["empty_desc"] >= 0.3)]
        .sort_values("n", ascending=False)
        .head(args.top)
        .iterrows()
    ):
        print(
            f"  [{r['cluster']:3d}] {r['region'][:60]:60} n={r['n']:3d} unscheduled {r['unscheduled']:.0%} empty {r['empty_desc']:.0%} | {r['depts']}"
        )

    print("\n── most department-diverse regions (content cutting across the org chart) ──")
    for _, r in reg.sort_values("n_depts", ascending=False).head(10).iterrows():
        print(f"  [{r['cluster']:3d}] {r['region'][:60]:60} n={r['n']:3d} {r['n_depts']} depts | {r['depts']}")

    print("\n── corpus-wide title-pattern counts ──")
    for name, pat in TITLE_PATTERNS.items():
        m = df["title"].str.contains(pat, case=False, regex=True)
        print(f"  {name:22} {m.sum():5d} courses; unlabelled share {(df.loc[m, 'cluster'] < 0).mean():.0%}")
    print("\n── most duplicated description prefixes corpus-wide ──")
    for k, v in Counter(df["description"].str[:100]).most_common(10):
        print(f"  {v:4d} | {k[:90]}")


if __name__ == "__main__":
    main()
