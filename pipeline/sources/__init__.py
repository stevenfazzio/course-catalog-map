"""University adapters.

Each module exposes:
    fetch_raw(paths)  -> writes the raw catalog under paths["raw"]; resumable, never re-downloads.
    parse_raw(paths)  -> (courses_df, listings_df, drops_df) in the common schema below.

Everything downstream of stage 01 reads courses.parquet and knows nothing about the source.
"""

import importlib

# One row per course (a registrar-level course identity, not a listing or a section).
COMMON_COLUMNS = [
    "course_id",  # source-specific stable id, unique per row
    "university",  # adapter name, e.g. "stanford"
    "codes",  # every catalog code the course is listed under, e.g. ["CS 224N", "LINGUIST 284"]
    "primary_code",  # the code used for display; first of `codes` in sorted order
    "subject",  # subject prefix of primary_code
    "title",
    "description",
    "school",  # top-level academic unit
    "department",  # long name of the department that owns primary_code
    "department_code",
    "career",  # UG / GR / other source-specific career codes
    "units_min",
    "units_max",
    "grading",
    "gers",  # general-education / breadth requirement tags
    "components",  # instructional formats seen in sections, e.g. ["LEC", "DIS"]
    "terms_offered",
    "instructors",
    "n_sections",
    "scheduled",  # at least one section scheduled in the catalog year
    "n_listings",  # how many codes the course was listed under
    "catalog_year",
    "url",
]


def get_source(name: str):
    return importlib.import_module(f"sources.{name}")
