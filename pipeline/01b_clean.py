"""Stage 01b: apply the preregistered corpus rules (docs/preregistration.md, part 2)
-> data/<source>/corpus.parquet, corpus_drops.csv, corpus_meta.json.

Rules, applied in order so that each course is counted under one rule:
  1. placeholder      description shorter than CORPUS_MIN_DESCRIPTION_WORDS words          -> dropped
  2. administrative   whole description shared verbatim (after normalisation) by at least
                      CORPUS_SHARED_MIN_COURSES courses across at least
                      CORPUS_SHARED_MIN_DEPARTMENTS departments                             -> dropped
  3. template         an opening run of whole sentences, at least CORPUS_TEMPLATE_MIN_WORDS words
                      long and shared by at least CORPUS_SHARED_MIN_COURSES courses, is removed
                      from the embedded text; if nothing remains, the title stands alone  -> kept
Every rule is written against course content or metadata, never map position. The hovercard keeps the
full catalog description; only `embed_text` changes.
"""

import argparse
import datetime as dt
import json
import re
from collections import Counter

import pandas as pd

import config
from io_utils import atomic_write_text, report_delta, validate_stage_output, write_parquet_safely

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", text.lower())).strip()


def sentence_prefixes(description: str) -> list[str]:
    """Cumulative runs of whole sentences from the start: the first sentence, the first two, ..."""
    sentences = [s for s in SENTENCE_SPLIT.split(description.strip()) if s]
    return [" ".join(sentences[:i]) for i in range(1, len(sentences) + 1)]


def apply_rules(courses: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = courses.copy()
    words = df["description"].str.split().str.len()
    drops = []

    # Rule 1: placeholders.
    r1 = words < config.CORPUS_MIN_DESCRIPTION_WORDS
    for _, r in df[r1].iterrows():
        drops.append(
            {
                "rule": "1_placeholder",
                "course_id": r["course_id"],
                "code": r["primary_code"],
                "title": r["title"],
                "description": r["description"][:80],
            }
        )
    df = df[~r1]

    # Rule 2: administrative boilerplate, whole description shared across departments.
    df = df.assign(_norm=df["description"].map(normalise))
    groups = df.groupby("_norm").agg(n=("course_id", "size"), n_dept=("department_code", "nunique"))
    admin = groups[
        (groups["n"] >= config.CORPUS_SHARED_MIN_COURSES) & (groups["n_dept"] >= config.CORPUS_SHARED_MIN_DEPARTMENTS)
    ]
    r2 = df["_norm"].isin(admin.index)
    for _, r in df[r2].iterrows():
        drops.append(
            {
                "rule": "2_administrative",
                "course_id": r["course_id"],
                "code": r["primary_code"],
                "title": r["title"],
                "description": r["description"][:80],
            }
        )
    df = df[~r2]

    # Rule 3: shared opening text. Count every sentence-aligned prefix among the kept courses, then strip
    # from each course the longest prefix that is both long enough and shared widely enough.
    prefixes = df["description"].map(sentence_prefixes)
    counts = Counter(normalise(p) for plist in prefixes for p in plist)
    rule, remainder, stripped_words, shared_by = [], [], [], []
    for desc, plist in zip(df["description"], prefixes):
        chosen = None
        for p in plist:  # ascending length; keep the longest that qualifies
            if (
                len(p.split()) >= config.CORPUS_TEMPLATE_MIN_WORDS
                and counts[normalise(p)] >= config.CORPUS_SHARED_MIN_COURSES
            ):
                chosen = p
        if chosen is None:
            rule.append("full"), remainder.append(desc), stripped_words.append(0), shared_by.append(0)
            continue
        rest = desc[len(chosen) :].strip()
        rule.append("template_stripped" if rest else "title_only")
        remainder.append(rest)
        stripped_words.append(len(chosen.split()))
        shared_by.append(counts[normalise(chosen)])
    df = df.assign(
        embed_text_rule=rule,
        embedded_description=remainder,
        stripped_words=stripped_words,
        template_shared_by=shared_by,
    )
    title = df["title"].str.strip()
    df["embed_text"] = title.where(df["embedded_description"] == "", title + "\n" + df["embedded_description"])
    df = df.drop(columns=["_norm"])

    drops_df = pd.DataFrame(drops, columns=["rule", "course_id", "code", "title", "description"])
    stats = {
        "rule_1_placeholder_dropped": int(r1.sum()),
        "rule_2_administrative_dropped": int(r2.sum()),
        "rule_2_groups": int(len(admin)),
        "rule_3_template_stripped": int((df["embed_text_rule"] == "template_stripped").sum()),
        "rule_3_title_only": int((df["embed_text_rule"] == "title_only").sum()),
        "kept": int(len(df)),
    }
    return df.reset_index(drop=True), drops_df, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    courses = pd.read_parquet(paths["courses"])
    corpus, drops, stats = apply_rules(courses)

    report_delta("01b_clean", len(courses), len(corpus), "rules 1-2 (see corpus_drops.csv)")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("  rule 3 most-stripped openings:")
    top = (
        corpus[corpus["stripped_words"] > 0]
        .groupby("stripped_words")
        .agg(n=("course_id", "size"), example=("embed_text_rule", "first"))
    )
    for w, r in top.sort_values("n", ascending=False).head(6).iterrows():
        print(f"    {r['n']:4d} courses lost a {w}-word opening ({r['example']})")
    validate_stage_output(corpus, "01b_clean", ["course_id", "embed_text", "embed_text_rule"])
    assert corpus["course_id"].is_unique and (corpus["embed_text"].str.strip() != "").all()

    write_parquet_safely(corpus, paths["corpus"])
    drops.to_csv(paths["corpus_drops"], index=False)
    meta = {
        "thresholds": {
            "min_description_words": config.CORPUS_MIN_DESCRIPTION_WORDS,
            "shared_min_courses": config.CORPUS_SHARED_MIN_COURSES,
            "shared_min_departments": config.CORPUS_SHARED_MIN_DEPARTMENTS,
            "template_min_words": config.CORPUS_TEMPLATE_MIN_WORDS,
        },
        "counts": stats,
        "n_in": int(len(courses)),
        "cleaned_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(paths["corpus_meta"], json.dumps(meta, indent=2))
    print(
        f"wrote {paths['corpus']} ({len(corpus)} rows), {paths['corpus_drops']} ({len(drops)} rows), corpus_meta.json"
    )


if __name__ == "__main__":
    main()
