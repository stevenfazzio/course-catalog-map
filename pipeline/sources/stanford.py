"""Stanford ExploreCourses adapter.

API shape, observed 2026-09-04:
  GET {base}/?view=xml-20200810
      -> <schools><school name="..."><department longname="..." name="CODE"/>...</school>...</schools>
  GET {base}/search?view=xml-20200810&academicYear=YYYYYYYY&filter-coursestatus-Active=on
                    &filter-departmentcode-CODE=on&q=CODE
      -> <xml><courses><course>...</course></courses></xml> for that department, sections inline.
         Sections are ~93% of the bytes; each department is one request, up to ~15 MB.

A cross-listed course appears once in the file of every department that lists it, with the
same administrativeInformation/courseId and that department's subject/code on each copy.
"""

import datetime as dt
import json
import time
import xml.etree.ElementTree as ET

import requests

import config
from io_utils import atomic_write_bytes, atomic_write_text, fetch_with_retry

INDEX_FILE = "_departments.xml"
MANIFEST_FILE = "_manifest.json"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    return s


def parse_department_index(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    depts = []
    for school in root.findall(".//school"):
        for dept in school.findall("department"):
            depts.append(
                {
                    "school": school.get("name", ""),
                    "department_code": dept.get("name", ""),
                    "department": dept.get("longname", ""),
                }
            )
    return depts


def unique_departments(depts: list[dict]) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Dedupe the index by department code, keeping the first long name.

    The index lists TAPS ("Drama" / "Theater and Performance Studies") and ILAC ("Iberian & Latin
    American Cultures" / "Spanish, Portuguese, & Catalan Literature") twice under the same school.
    They are one department each; the search endpoint returns the same file for both entries.
    """
    seen: dict[str, dict] = {}
    aliases = []
    for d in depts:
        code = d["department_code"]
        if code in seen:
            aliases.append((code, seen[code]["department"], d["department"]))
        else:
            seen[code] = d
    return sorted(seen.values(), key=lambda d: d["department_code"]), aliases


def _count_courses(xml_bytes: bytes) -> int:
    """Parse check before the file is committed to disk: malformed XML raises here."""
    return len(ET.fromstring(xml_bytes).findall(".//course"))


def fetch_raw(paths: dict) -> None:
    raw = paths["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    session = _session()

    index_path = raw / INDEX_FILE
    if not index_path.exists():
        body = fetch_with_retry(session, config.STANFORD_BASE_URL, params={"view": config.STANFORD_XML_VIEW})
        assert b"<department" in body, "department index did not look like the schools/departments XML"
        atomic_write_bytes(index_path, body)
    depts, _ = unique_departments(parse_department_index(index_path.read_bytes()))

    todo = [d for d in depts if not (raw / f"{d['department_code']}.xml").exists()]
    print(f"{len(depts)} departments: {len(depts) - len(todo)} cached, {len(todo)} to fetch")

    for n, d in enumerate(todo, start=1):
        code = d["department_code"]
        params = {
            "view": config.STANFORD_XML_VIEW,
            "academicYear": config.STANFORD_ACADEMIC_YEAR,
            "filter-coursestatus-Active": "on",
            f"filter-departmentcode-{code}": "on",
            "q": code,
        }
        body = fetch_with_retry(session, config.STANFORD_BASE_URL + "search", params=params)
        n_courses = _count_courses(body)
        atomic_write_bytes(raw / f"{code}.xml", body)
        print(f"[{n}/{len(todo)}] {code}: {n_courses} courses, {len(body) / 1e6:.1f} MB", flush=True)
        time.sleep(config.REQUEST_DELAY_S)

    manifest = {
        "source": "stanford",
        "academic_year": config.STANFORD_ACADEMIC_YEAR,
        "xml_view": config.STANFORD_XML_VIEW,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_departments": len(depts),
        "course_status_filter": "Active",
    }
    atomic_write_text(raw / MANIFEST_FILE, json.dumps(manifest, indent=2))
    print(f"manifest written: {raw / MANIFEST_FILE}")


# ── Parse ────────────────────────────────────────────────────────────────────

import re  # noqa: E402
from collections import Counter  # noqa: E402

import pandas as pd  # noqa: E402

# A catalog code as it appears inside a title's cross-listing suffix: "LINGUIST 284", "MS&E 231", "CS 129X".
_CODE_RE = r"[A-Z][A-Z&\-]*\s?\d+[A-Z]*"
# ExploreCourses appends the other listings to the title: "Natural Language Processing (LINGUIST 284, SYMSYS 195N)".
# That is organisational metadata, not content, so it is stripped from `title` and kept in `title_raw`.
_XLIST_SUFFIX_RE = re.compile(r"\s*\(\s*" + _CODE_RE + r"(?:\s*,\s*" + _CODE_RE + r")*\s*\)\s*$")


def _t(el, path: str) -> str:
    x = el.findtext(path) if el is not None else None
    return " ".join(x.split()) if x else ""


def _float(s: str):
    try:
        return float(s) if s else None
    except ValueError:
        return None


def course_url(subject: str, code: str) -> str:
    return (
        f"{config.STANFORD_BASE_URL}search?view=catalog&academicYear={config.STANFORD_ACADEMIC_YEAR}"
        f"&q={subject}{code}&filter-departmentcode-{subject}=on&filter-coursestatus-Active=on"
    )


def _listing(el, dept: dict) -> dict:
    admin = el.find("administrativeInformation")
    sections = el.findall("sections/section")
    subject, code = _t(el, "subject"), _t(el, "code")
    title_raw = _t(el, "title")
    return {
        "course_id": _t(admin, "courseId"),
        "offer_number": int(_t(admin, "offerNumber") or 0),
        "listed_in": dept["department_code"],
        "school": dept["school"],
        "department": dept["department"],
        "subject": subject,
        "code": code,
        "listing_code": f"{subject} {code}",
        "title_raw": title_raw,
        "title": _XLIST_SUFFIX_RE.sub("", title_raw),
        "description": _t(el, "description"),
        "gers": [g.strip() for g in _t(el, "gers").split(",") if g.strip()],
        "repeatable": _t(el, "repeatable") == "true",
        "grading": _t(el, "grading"),
        "units_min": _float(_t(el, "unitsMin")),
        "units_max": _float(_t(el, "unitsMax")),
        "catalog_year": _t(el, "year"),
        "academic_group": _t(admin, "academicGroup"),
        "academic_organization": _t(admin, "academicOrganization"),
        "career": _t(admin, "academicCareer"),
        "effective_status": _t(admin, "effectiveStatus"),
        "catalog_print": _t(admin, "catalogPrint"),
        "final_exam": _t(admin, "finalExamFlag"),
        "learning_objectives": [_t(lo, "description") for lo in el.findall("learningObjectives/learningObjective")],
        "quarters_attr": sorted(
            {_t(a, "value") for a in el.findall("attributes/attribute") if _t(a, "name") == "NQTR"} - {""}
        ),
        "tags": sorted({f"{_t(t, 'organization')}:{_t(t, 'name')}" for t in el.findall("tags/tag")}),
        "components": sorted({_t(s, "component") for s in sections} - {""}),
        "terms_offered": sorted({_t(s, "term") for s in sections} - {""}),
        "instructors": sorted(
            {_t(i, "name") for s in sections for i in s.findall("schedules/schedule/instructors/instructor")} - {""}
        ),
        "n_sections": len(sections),
    }


def parse_listings(paths: dict, only: list[str] | None = None) -> pd.DataFrame:
    """One row per <course> element per department file, i.e. one row per catalog listing."""
    raw = paths["raw"]
    depts, aliases = unique_departments(parse_department_index((raw / INDEX_FILE).read_bytes()))
    for code, kept, dropped in aliases:
        print(f"department index lists {code} twice; using {kept!r}, ignoring alias {dropped!r}")
    if only:
        depts = [d for d in depts if d["department_code"] in set(only)]
    rows = []
    for d in depts:
        path = raw / f"{d['department_code']}.xml"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run 00_fetch.py first")
        for el in ET.parse(path).getroot().findall(".//course"):
            rows.append(_listing(el, d))
    return pd.DataFrame(rows)


def _home_department_map(listings: pd.DataFrame) -> dict[str, set[str]]:
    """academic_organization -> the department codes it owns, learned from single-listing courses.

    ExploreCourses gives every copy of a cross-listed course the owning unit's academicOrganization
    (e.g. COMPUTSCI) but not that unit's department code (CS). Courses listed in exactly one department
    tie the two together. One organization can own several codes: ASIANLANG owns CHINA, JAPAN and KOREA.
    """
    n_listings = listings.groupby("course_id")["course_id"].transform("size")
    single = listings[n_listings == 1]
    return {org: set(grp["listed_in"]) for org, grp in single.groupby("academic_organization") if org}


def collapse_listings(listings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Collapse listings to one row per registrar courseId.

    Primary listing = one in a department the course's academic_organization owns, then the lowest
    offerNumber, then the alphabetically first code. All codes are kept in `codes`.
    """
    org_to_depts = _home_department_map(listings)
    stats = Counter()
    courses, dropped = [], []
    for course_id, grp in listings.groupby("course_id", sort=True):
        home = org_to_depts.get(grp["academic_organization"].iloc[0], set())
        grp = grp.assign(_is_home=grp["listed_in"].isin(home)).sort_values(
            ["_is_home", "offer_number", "listing_code"], ascending=[False, True, True]
        )
        primary = grp.iloc[0]
        if len(grp) > 1:
            stats["cross_listed_courses"] += 1
            stats["primary_by_home_department" if primary["_is_home"] else "primary_by_fallback"] += 1
            if grp["description"].nunique() > 1:
                stats["description_differs_across_listings"] += 1
            if grp["title"].nunique() > 1:
                stats["title_differs_across_listings"] += 1
            for _, other in grp.iloc[1:].iterrows():
                dropped.append(
                    {
                        "course_id": course_id,
                        "listing_code": other["listing_code"],
                        "listed_in": other["listed_in"],
                        "rule": "collapsed_cross_listing",
                        "kept_as": primary["listing_code"],
                    }
                )
        codes = sorted(set(grp["listing_code"]))
        row = primary.drop(labels=["_is_home", "listing_code", "code", "offer_number", "listed_in"]).to_dict()
        row.update(
            {
                "university": "stanford",
                "codes": codes,
                "primary_code": primary["listing_code"],
                "department_code": primary["listed_in"],
                "n_listings": len(codes),
                "offer_numbers": sorted(set(grp["offer_number"])),
                "listed_in": sorted(set(grp["listed_in"])),
                "units_min": grp["units_min"].min(),
                "units_max": grp["units_max"].max(),
                "components": sorted(set().union(*grp["components"])),
                "terms_offered": sorted(set().union(*grp["terms_offered"])),
                "instructors": sorted(set().union(*grp["instructors"])),
                "n_sections": int(grp["n_sections"].max()),
                "scheduled": bool(grp["n_sections"].max() > 0),
                "url": course_url(primary["subject"], primary["code"]),
            }
        )
        courses.append(row)
    return pd.DataFrame(courses), pd.DataFrame(dropped), dict(stats)


def parse_raw(paths: dict, only: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    listings = parse_listings(paths, only=only)
    courses, drops, stats = collapse_listings(listings)
    print(
        f"stanford: {len(listings)} listings -> {len(courses)} courses "
        f"({len(drops)} listings collapsed as cross-listings)"
    )
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    return courses, drops, listings


# ── Prompt context for Toponymy's namer (stage 04) ───────────────────────────
OBJECT_DESCRIPTION = "university course listings (title and catalog description)"
CORPUS_DESCRIPTION = "the complete 2026-27 course catalog of Stanford University, across every school and department"
