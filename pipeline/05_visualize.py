"""Stage 05: render the interactive map -> data/<source>/<source>_course_map.html.

Reads courses.parquet, umap_coords.npz and labels.parquet; everything here is source-agnostic.
"""

import argparse
import html
import os

import datamapplot
import glasbey
import numpy as np
import pandas as pd

import config

# Stanford section components -> readable format. Unknown codes pass through unchanged.
COMPONENT_NAMES = {
    "LEC": "Lecture",
    "SEM": "Seminar",
    "COL": "Colloquium",
    "WKS": "Workshop",
    "LAB": "Laboratory",
    "LBS": "Lab section",
    "DIS": "Discussion",
    "PRA": "Practicum",
    "ACT": "Activity",
    "CLK": "Clerkship",
    "T/D": "Thesis / dissertation",
    "INS": "Independent study",
    "ISF": "Independent study",
}
# When a course has several section types, the first of these present is its primary format.
FORMAT_PRIORITY = ["LEC", "SEM", "COL", "WKS", "LAB", "LBS", "DIS", "PRA", "ACT", "CLK", "T/D", "INS", "ISF"]
CAREER_NAMES = {"UG": "Undergraduate", "GR": "Graduate", "LAW": "Law", "GSB": "Business (GSB)", "MED": "Medicine (MD)"}
DESCRIPTION_HOVER_CHARS = 480

HOVER_TEMPLATE = (
    "<div style=\"font-family:'IBM Plex Sans',sans-serif;width:min(400px,100%);padding:8px 10px;"
    'box-sizing:border-box;color:#1f2328;">'
    '<div style="font-size:11.5px;color:#57606a;">{school_dept}</div>'
    '<div style="font-size:14px;font-weight:600;line-height:1.3;margin:2px 0 6px;">'
    '<span style="color:#8b949e;font-weight:500;">{code}</span> {title}</div>'
    '<div style="font-size:11.5px;color:#57606a;margin-bottom:8px;">{facts}</div>'
    '<div style="font-size:13px;line-height:1.45;margin-bottom:8px;">{description}</div>'
    '<div style="font-size:11px;color:#8b949e;line-height:1.6;">{footer}</div>'
    "</div>"
)

CUSTOM_CSS = """
.deck-tooltip { max-width: min(440px, 92vw) !important; }
/* DataMapPlot sizes the dropdown swatch to its colour count (12px per box, up to five), so a two-category
   colormap gets a 24px swatch and its label sits 36px left of the others. Fix the swatch width and let
   the boxes share it. */
.color-swatch { display: inline-flex !important; width: 60px !important; }
.color-swatch .color-swatch-box { flex: 1 1 0 !important; width: auto !important; }
/* Legend rows (.legend-item > .color-swatch-box) are not inside .color-swatch and keep their 12px boxes. */
"""
CUSTOM_JS = """
datamap.deckgl.setProps({controller: {scrollZoom: {speed: 0.05, smooth: true}}});
"""


def categorical_palette(n_colors: int) -> list[str]:
    return glasbey.create_palette(
        palette_size=n_colors,
        colorblind_safe=True,
        cvd_severity=100.0,
        lightness_bounds=(25, 75),  # no swatch washes out on a light background
    )


def categorical(
    field: str, description: str, values: np.ndarray, neutral: str | None = None
) -> tuple[dict, np.ndarray]:
    """Colormap metadata for a categorical field; `neutral` names a category pinned to grey."""
    cats = sorted(set(values.tolist()) - {neutral})
    mapping = dict(zip(cats, categorical_palette(len(cats))))
    if neutral is not None and neutral in set(values.tolist()):
        mapping[neutral] = "#bdbdbd"
    meta = {
        "field": field,
        "description": description,
        "kind": "categorical",
        "color_mapping": mapping,
        "show_legend": True,
    }
    return meta, values


def primary_format(components: list[str]) -> str:
    for code in FORMAT_PRIORITY:
        if code in components:
            return COMPONENT_NAMES[code]
    return COMPONENT_NAMES.get(components[0], components[0]) if len(components) else "No sections this year"


def primary_ger(gers: list[str]) -> str:
    ways = [g for g in gers if g.startswith("WAY-")]
    return (ways or list(gers) or ["None"])[0]


def esc(s) -> str:
    return html.escape(str(s) if s is not None else "")


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def load_label_layers(files: dict, course_ids: np.ndarray) -> list[np.ndarray]:
    labels = pd.read_parquet(files["labels"])
    assert (labels["course_id"].to_numpy() == course_ids).all(), "labels.parquet is not aligned with courses.parquet"
    cols = sorted((c for c in labels.columns if c.startswith("label_layer_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    layers = [labels[c].fillna("Unlabelled").to_numpy() for c in cols]
    n_finest, n_coarsest = pd.Series(layers[0]).nunique(), pd.Series(layers[-1]).nunique()
    assert n_finest >= n_coarsest, "label layers must be finest-first for DataMapPlot"
    print(f"{len(layers)} label layers: " + ", ".join(f"{pd.Series(v).nunique() - 1} named" for v in layers))
    return layers


def build_point_data(courses: pd.DataFrame) -> pd.DataFrame:
    units = np.where(
        courses["units_min"] == courses["units_max"],
        courses["units_max"].map(lambda u: f"{u:g} units" if pd.notna(u) else ""),
        courses["units_min"].map(lambda u: f"{u:g}" if pd.notna(u) else "?")
        + "–"
        + courses["units_max"].map(lambda u: f"{u:g} units" if pd.notna(u) else "?"),
    )
    fmt = courses["components"].map(primary_format)
    terms = courses["terms_offered"].map(
        lambda t: ", ".join(x.split()[-1] for x in t) if len(t) else "not scheduled 2026–27"
    )
    facts = [
        " · ".join(p for p in (CAREER_NAMES.get(c, c), u, f, t) if p)
        for c, u, f, t in zip(courses["career"], units, fmt, terms)
    ]
    crosslist = courses["codes"].map(lambda cs: "Also listed as " + ", ".join(cs[1:]) if len(cs) > 1 else "")
    gers = courses["gers"].map(lambda g: ", ".join(g))
    footer = [" &nbsp;·&nbsp; ".join(p for p in (esc(x), esc(g)) if p) for x, g in zip(crosslist, gers)]
    search = (
        courses["codes"].map(" ".join)
        + " "
        + courses["title"]
        + " "
        + courses["description"]
        + " "
        + courses["department"]
        + " "
        + courses["school"]
        + " "
        + courses["instructors"].map(" ".join)
    )
    return pd.DataFrame(
        {
            "code": courses["primary_code"].map(esc),
            "title": courses["title"].map(esc),
            "school_dept": [esc(f"{s} · {d}") for s, d in zip(courses["school"], courses["department"])],
            "facts": [esc(f) for f in facts],
            "description": courses["description"].map(lambda d: esc(truncate(d, DESCRIPTION_HOVER_CHARS))),
            "footer": footer,
            "url": courses["url"],
            "search": search,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--raw", action="store_true", help="render the raw-catalog build instead")
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    files = config.keyed_files(paths, args.embedding, raw=args.raw)

    courses = pd.read_parquet(paths["courses"] if args.raw else paths["corpus"])
    ids = courses["course_id"].to_numpy()
    lay = np.load(files["umap"], allow_pickle=True)
    assert (lay["course_id"] == ids).all(), "umap_coords.npz is not aligned with courses.parquet"
    coords = lay["coords"]
    label_layers = load_label_layers(files, ids)
    extra = build_point_data(courses)
    hover_text = (courses["primary_code"] + " " + courses["title"]).tolist()

    school_meta, school_vals = categorical("school", "School", courses["school"].to_numpy())
    career_meta, career_vals = categorical(
        "career", "Level", courses["career"].map(lambda c: CAREER_NAMES.get(c, c)).to_numpy()
    )
    sched_meta, sched_vals = categorical(
        "scheduled",
        "Scheduled in 2026–27",
        np.where(courses["scheduled"], "Scheduled", "Not scheduled"),
        neutral="Not scheduled",
    )
    format_meta, format_vals = categorical(
        "format",
        "Primary format",
        courses["components"].map(primary_format).to_numpy(),
        neutral="No sections this year",
    )
    ger_meta, ger_vals = categorical(
        "ger", "General-education requirement (WAYS)", courses["gers"].map(primary_ger).to_numpy(), neutral="None"
    )
    units_vals = courses["units_max"].fillna(0).astype(float).to_numpy()
    units_meta = {"field": "units", "description": "Units (max)", "kind": "continuous", "cmap": "viridis"}

    plot = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=hover_text,
        hover_text_html_template=HOVER_TEMPLATE,
        extra_point_data=extra,
        on_click="window.open(`{url}`, `_blank`)",
        enable_search=True,
        search_field="search",
        title="Stanford Course Map",
        sub_title=(
            f"{len(courses):,} courses in the 2026–27 catalog, positioned by the meaning of their descriptions"
            + ("" if args.embedding == config.EMBED_MODEL_KEY else f" · exploration build: {args.embedding}")
            + (" · raw catalog" if args.raw else "")
        ),
        font_family="IBM Plex Sans",
        cvd_safer=True,
        noise_label="Unlabelled",
        initial_zoom_fraction=config.MAP_INITIAL_ZOOM_FRACTION,
        colormap_rawdata=[school_vals, career_vals, sched_vals, format_vals, ger_vals, units_vals],
        colormap_metadata=[school_meta, career_meta, sched_meta, format_meta, ger_meta, units_meta],
        custom_css=CUSTOM_CSS,
        custom_js=CUSTOM_JS,
        inline_data=True,
    )
    plot.save(str(files["map_html"]))
    size_mb = os.stat(files["map_html"]).st_size / 1e6
    print(f"wrote {files['map_html']} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
