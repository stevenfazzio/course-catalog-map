import importlib

import pandas as pd

clean = importlib.import_module("01b_clean")

ADMIN = (
    "Students undertake investigations sponsored by individual faculty members. Prerequisite: consent of instructor."
)
TEMPLATE = (
    "PWR 1 courses focus on developing writing and revision strategies for rhetorical analysis "
    "and research-based arguments."
)


def _course(i, dept, title, desc):
    return {
        "course_id": str(i),
        "primary_code": f"{dept} {i}",
        "department_code": dept,
        "title": title,
        "description": desc,
    }


def test_rules_apply_in_order_and_count_once():
    rows = []
    rows.append(_course(1, "CS", "TGR Dissertation", ""))  # rule 1: empty
    rows.append(_course(2, "EE", "Thesis", "TGR Dissertation"))  # rule 1: two words
    for i, dept in enumerate(["A", "B", "C", "D", "E"], start=10):  # rule 2: shared across 5 departments
        rows.append(_course(i, dept, "Undergraduate Research", ADMIN))
    for i in range(20, 26):  # rule 3: shared 16-word opening within one subject, distinct remainders
        rows.append(
            _course(
                i, "PWR", f"Writing & Rhetoric 1: Theme {i}", f"{TEMPLATE} This seminar takes theme {i} as its subject."
            )
        )
    for i in range(30, 35):  # rule 3: whole description shared within one subject -> title only
        rows.append(
            _course(
                i,
                "MUSIC",
                f"Private Lessons: Instrument {i}",
                "For Music majors and minors only. May be repeated for credit a total of 15 times by audition.",
            )
        )
    rows.append(
        _course(
            40, "HIST", "Modern Europe", "A survey of European history since 1789. Lectures and discussion sections."
        )
    )
    for i in range(50, 56):  # a short shared opening (4 words) must NOT be stripped
        rows.append(
            _course(
                i,
                "GER",
                f"Directed Reading {i}",
                f"Prerequisite: consent of instructor. Topic {i} chosen with the instructor.",
            )
        )
    df = pd.DataFrame(rows)
    corpus, drops, stats = clean.apply_rules(df)

    assert stats["rule_1_placeholder_dropped"] == 2
    assert stats["rule_2_administrative_dropped"] == 5 and stats["rule_2_groups"] == 1
    assert stats["rule_3_template_stripped"] == 6 and stats["rule_3_title_only"] == 5
    assert set(drops["rule"]) == {"1_placeholder", "2_administrative"} and len(drops) == 7
    assert stats["kept"] == len(df) - 7
    pwr = corpus[corpus["department_code"] == "PWR"].iloc[0]
    assert pwr["embed_text"] == f"{pwr['title']}\nThis seminar takes theme 20 as its subject."
    assert pwr["stripped_words"] == 16 and pwr["template_shared_by"] == 6
    music = corpus[corpus["department_code"] == "MUSIC"].iloc[0]
    assert music["embed_text"] == music["title"] and music["embed_text_rule"] == "title_only"
    hist = corpus[corpus["department_code"] == "HIST"].iloc[0]
    assert hist["embed_text_rule"] == "full" and hist["embed_text"].endswith("discussion sections.")
    ger = corpus[corpus["department_code"] == "GER"].iloc[0]
    assert ger["embed_text_rule"] == "full"  # 4-word shared opening is below the template threshold


def test_sentence_prefixes_are_cumulative():
    assert clean.sentence_prefixes("One. Two! Three?") == ["One.", "One. Two!", "One. Two! Three?"]
    assert clean.sentence_prefixes("") == []


def test_normalise_ignores_case_punctuation_and_spacing():
    assert clean.normalise("Prerequisite:  Consent of instructor.") == clean.normalise(
        "prerequisite consent of instructor"
    )
