import pandas as pd
import pytest

from sources import stanford


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            "Natural Language Processing with Deep Learning (LINGUIST 284, SYMSYS 195N)",
            "Natural Language Processing with Deep Learning",
        ),
        (
            "Computers, Ethics, and Public Policy (WIM)",
            "Computers, Ethics, and Public Policy (WIM)",
        ),  # WIM is content, not a code
        (
            "Perspectives in Assistive Technology (ENGR 110) (ENGR 210)",
            "Perspectives in Assistive Technology (ENGR 110)",
        ),
        (
            "Gender and Politics in the Middle East (19th and 20th Centuries)",
            "Gender and Politics in the Middle East (19th and 20th Centuries)",
        ),
        ("Optimization (MS&E 211)", "Optimization"),
        ("Plain Title", "Plain Title"),
    ],
)
def test_cross_listing_suffix_is_stripped(raw, expected):
    assert stanford._XLIST_SUFFIX_RE.sub("", raw) == expected


def test_unique_departments_keeps_first_long_name():
    depts = [
        {"school": "H&S", "department_code": "TAPS", "department": "Drama"},
        {"school": "H&S", "department_code": "CS", "department": "Computer Science"},
        {"school": "H&S", "department_code": "TAPS", "department": "Theater and Performance Studies"},
    ]
    unique, aliases = stanford.unique_departments(depts)
    assert [d["department_code"] for d in unique] == ["CS", "TAPS"]
    assert unique[1]["department"] == "Drama"
    assert aliases == [("TAPS", "Drama", "Theater and Performance Studies")]


def _listing(course_id, listed_in, subject, code, org, offer=1, **extra):
    row = {
        "course_id": course_id,
        "offer_number": offer,
        "listed_in": listed_in,
        "school": "S",
        "department": listed_in,
        "subject": subject,
        "code": code,
        "listing_code": f"{subject} {code}",
        "title_raw": "T",
        "title": "T",
        "description": "D",
        "gers": [],
        "repeatable": False,
        "grading": "Letter",
        "units_min": 3.0,
        "units_max": 3.0,
        "catalog_year": "2026-2027",
        "academic_group": "G",
        "academic_organization": org,
        "career": "UG",
        "effective_status": "A",
        "catalog_print": "Y",
        "final_exam": "N",
        "learning_objectives": [],
        "quarters_attr": [],
        "tags": [],
        "components": [],
        "terms_offered": [],
        "instructors": [],
        "n_sections": 0,
        "navigator_term_id": "",
        "navigator_class_id": "",
    }
    row.update(extra)
    return row


def test_collapse_prefers_home_department_and_keeps_all_codes():
    listings = pd.DataFrame(
        [
            _listing("1", "CS", "CS", "101", "COMPUTSCI"),  # single listing: teaches COMPUTSCI -> CS
            _listing("2", "LINGUIST", "LINGUIST", "284", "COMPUTSCI", n_sections=2, components=["LEC"]),
            _listing(
                "2",
                "CS",
                "CS",
                "224N",
                "COMPUTSCI",
                n_sections=2,
                components=["LEC", "DIS"],
                navigator_term_id="1274",
                navigator_class_id="1953",
            ),
            _listing("3", "CHINA", "CHINA", "111", "ASIANLANG", offer=1),  # no single-listing course for ASIANLANG
            _listing("3", "CHINA", "CHINA", "211", "ASIANLANG", offer=2),
        ]
    )
    courses, drops, stats = stanford.collapse_listings(listings)
    by_id = courses.set_index("course_id")
    assert by_id.loc["2", "primary_code"] == "CS 224N"
    assert by_id.loc["2", "codes"] == ["CS 224N", "LINGUIST 284"]
    assert by_id.loc["2", "components"] == ["DIS", "LEC"]
    assert bool(by_id.loc["2", "scheduled"])
    assert by_id.loc["3", "primary_code"] == "CHINA 111"  # fallback: lowest offer number
    assert by_id.loc["1", "n_listings"] == 1
    assert stats == {"cross_listed_courses": 2, "primary_by_home_department": 1, "primary_by_fallback": 1}
    assert set(drops["listing_code"]) == {"LINGUIST 284", "CHINA 211"}
    assert courses["course_id"].is_unique
    # Scheduled courses link to Navigator's class page; unscheduled ones fall back to the ExploreCourses catalog.
    assert by_id.loc["2", "url"] == "https://navigator.stanford.edu/classes/1274/1953"
    assert by_id.loc["1", "url"].startswith("https://explorecourses.stanford.edu/search?view=catalog")
    assert by_id.loc["2", "url_catalog"].startswith("https://explorecourses.stanford.edu/")


def test_first_class_picks_earliest_term():
    import xml.etree.ElementTree as ET

    xml = ET.fromstring(
        "<sections><section><termId>1274</termId><classId>9</classId></section>"
        "<section><termId>1272</termId><classId>7</classId></section></sections>"
    )
    assert stanford._first_class(xml.findall("section")) == ("1272", "7")
    assert stanford._first_class([]) == ("", "")
