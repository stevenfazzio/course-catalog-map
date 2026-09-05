"""Stage 01: parse one university's raw catalog into the common schema.

Writes data/<source>/listings.parquet (one row per catalog listing, pre-collapse),
data/<source>/courses.parquet (one row per course) and data/<source>/drops.csv (every listing
removed by the collapse, with the rule and the row it was folded into).
"""

import argparse

import config
from io_utils import report_delta, validate_stage_output, write_parquet_safely
from sources import COMMON_COLUMNS, get_source


def profile(courses) -> None:
    print("\n── profile ──")
    print("by school:\n" + courses["school"].value_counts().to_string())
    print("by career:", courses["career"].value_counts().to_dict())
    words = courses["description"].str.split().str.len()
    print(
        f"description words: median {words.median():.0f}, p10 {words.quantile(0.1):.0f}, "
        f"p90 {words.quantile(0.9):.0f}, empty {(words == 0).sum()}"
    )
    print("no sections this year:", int((courses["n_sections"] == 0).sum()))
    print("cross-listed (n_listings>1):", int((courses["n_listings"] > 1).sum()))
    dup = courses["description"].str[:120].value_counts()
    print("most duplicated description prefixes:")
    for k, v in dup.head(8).items():
        print(f"  {v:4d} | {k[:100]}")
    print("titles still ending in a parenthetical:", int(courses["title"].str.contains(r"\)\s*$").sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--only", help="comma-separated department codes, for testing on a subset (no files written)")
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    only = args.only.split(",") if args.only else None

    courses, drops, listings = get_source(args.source).parse_raw(paths, only=only)
    validate_stage_output(courses, "01_parse", COMMON_COLUMNS)
    report_delta("01_parse", len(listings), len(courses), "cross-listed copies collapsed onto one courseId")
    assert courses["course_id"].is_unique
    profile(courses)

    if only:
        print("\n--only given: nothing written")
        return
    write_parquet_safely(listings, paths["listings"])
    write_parquet_safely(courses, paths["courses"])
    drops.to_csv(paths["drops"], index=False)
    print(f"\nwrote {paths['courses']} ({len(courses)} rows), {paths['listings']}, {paths['drops']}")


if __name__ == "__main__":
    main()
