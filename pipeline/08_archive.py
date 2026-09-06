"""Stage 08: archive the map's record on Zenodo with a DOI (docs/plan.md step 5).

What the deposit holds: the cleaned corpus and the catalog it was cut from, every candidate model's embeddings,
the published layout, labels and names, the exploration-map and comparison artifacts as one tarball, the versioned
record directory (docs/record/<source>/) as another, and the code at the archived commit. What it does not hold:
the raw ExploreCourses XML. ExploreCourses' footer links Stanford's site terms of use, which allow downloads "only
for User's own personal, non-commercial use" and forbid otherwise copying or distributing material, and its About
page describes the course data API as "available to students and faculty" (TERMS below). The snapshot stays local
and the deposit carries its per-file SHA-256 checksums instead, so a copy obtained from Stanford can be checked
against the one the map was built from.

Actions, in order. Each rewrites data/<source>/archive.json, which stage 07 copies into the record:
  --checksums   hash the raw snapshot into raw_checksums.json and hash the deposit's data files; no network
  --reserve     create the Zenodo deposition with the metadata and a pre-reserved DOI (or update its metadata)
  --upload      build the tarballs, the source zip and the README, upload every file, check Zenodo's MD5 of each
  --publish     publish the deposition; a published Zenodo record cannot be deleted, only versioned
--sandbox targets sandbox.zenodo.org with ZENODO_SANDBOX_TOKEN (separate accounts). --dry-run prints what would
be sent. Run stage 07 after --reserve, so the record tarball that --upload builds carries the DOI, and again
after --publish, so the committed record carries the published state.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import date, datetime, timezone
from pathlib import Path

import requests

import config
from io_utils import atomic_write_text

TERMS = {
    "read_on": "2026-09-06",
    "terms_url": "https://www.stanford.edu/site/terms/",
    "terms_quote_download": (
        "User may download material from the Sites only for User's own personal, non-commercial use."
    ),
    "terms_quote_copy": (
        "User may not otherwise copy, reproduce, retransmit, distribute, publish, commercially exploit or otherwise "
        "transfer any material."
    ),
    "about_url": "https://explorecourses.stanford.edu/about",
    "about_quote": (
        "A course data API is available to students and faculty. This API allows developers to programmatically "
        "query the course database."
    ),
    "how_read": (
        "ExploreCourses returned HTTP 503 on 2026-09-06. Its footer (Terms of Use -> stanford.edu/site/terms.html) "
        "and its About page were read from the Internet Archive capture of 2026-08-29; the Stanford terms page "
        "was read live."
    ),
    "decision": (
        "The raw XML snapshot is not redistributed. Its per-file SHA-256 checksums are deposited so that a copy "
        "obtained from Stanford can be verified against the one the map was built from. The corpus and catalog "
        "parquet files carry the course titles and descriptions the public map already displays; they are the "
        "minimum needed to audit the corpus rules and rebuild the embeddings."
    ),
}

# Deposited as individual files: the primary artifacts a reader wants without untarring, and the large ones.
TOP_LEVEL = [
    "corpus.parquet",
    "courses.parquet",
    "labels.parquet",
    "umap_coords.npz",
    "topic_names.json",
    "cluster_tree.json",
    "raw_checksums.json",
]
TOP_LEVEL_GLOBS = ["embeddings*.npz"]  # every candidate model, cleaned corpus and raw catalog
# Bundled as exploration_<source>.tar.gz: the two raw-catalog exploration maps' labels, names and layouts, the other
# candidates' layouts and run records, and the EVoC layers behind docs/embedding_structure.html. The trailing
# underscore in the layout and metadata patterns keeps the map's own files (in the record) out of the bundle.
EXPLORATION_GLOBS = [
    "labels_raw*.parquet",
    "labels_meta_raw*.json",
    "topic_names_raw*.json",
    "cluster_tree_raw*.json",
    "umap_coords_*.npz",
    "umap_meta_*.json",
    "embeddings_meta_*.json",
    "evoc_layers*.npz",
    "evoc_labels*.parquet",
]
KEYWORDS = [
    "course catalog",
    "curriculum",
    "higher education",
    "Stanford University",
    "text embeddings",
    "UMAP",
    "topic labelling",
    "data map",
]


# ── Hashing and file selection ────────────────────────────────────────────────
def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_checksums(raw_dir: Path) -> dict:
    """Per-file SHA-256 of the withheld snapshot, with the fetch manifest, so an independent copy can be verified."""
    files = sorted(p for p in raw_dir.iterdir() if p.is_file())
    rows = [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]
    manifest_path = raw_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    return {
        "n_files": len(rows),
        "total_bytes": sum(r["bytes"] for r in rows),
        "manifest": manifest,
        "terms": TERMS,
        "files": rows,
    }


def _globbed(base: Path, names: list[str], globs: list[str]) -> list[Path]:
    found = {base / n for n in names if (base / n).exists()}
    for pattern in globs:
        found.update(p for p in base.glob(pattern) if p.is_file())
    return sorted(found)


def deposit_data_files(base: Path) -> list[Path]:
    return _globbed(base, TOP_LEVEL, TOP_LEVEL_GLOBS)


def exploration_files(base: Path) -> list[Path]:
    return _globbed(base, [], EXPLORATION_GLOBS)


def make_tarball(out: Path, files: list[Path], prefix: str) -> None:
    """gzip tarball of `files` under `prefix/`, members sorted by name so the archive is reproducible."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for p in sorted(files, key=lambda q: q.name):
            tar.add(p, arcname=f"{prefix}/{p.name}", recursive=False)


# ── Git ──────────────────────────────────────────────────────────────────────
def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=config.ROOT, check=True, capture_output=True, text=True).stdout.strip()


def git_commit() -> str:
    return git("rev-parse", "HEAD")


def git_dirty() -> bool:
    return bool(git("status", "--porcelain"))


def git_archive(out: Path, prefix: str) -> None:
    subprocess.run(
        ["git", "archive", "--format=zip", f"--prefix={prefix}/", "-o", str(out), "HEAD"], cwd=config.ROOT, check=True
    )


# ── Metadata and text ─────────────────────────────────────────────────────────
def load_metas(base: Path) -> dict:
    metas = {}
    for key, name in [
        ("manifest", "raw/_manifest.json"),
        ("corpus", "corpus_meta.json"),
        ("embeddings", "embeddings_meta.json"),
        ("umap", "umap_meta.json"),
        ("labels", "labels_meta.json"),
        ("raw", "raw_checksums.json"),
    ]:
        p = base / name
        metas[key] = json.loads(p.read_text()) if p.exists() else {}
    return metas


def describe(source: str, metas: dict, commit: str, doi: str | None) -> list[str]:
    """The deposit's description as plain-text paragraphs: HTML for Zenodo, Markdown for the README."""
    counts = metas["corpus"].get("counts", {})
    layers = metas["labels"].get("layers", [])
    regions = " / ".join(str(layer["n_clusters"]) for layer in layers) if layers else "several"
    fetched = (metas["manifest"].get("fetched_at") or "")[:10]
    year = metas["manifest"].get("academic_year", "")
    year_text = f"{year[:4]}-{year[6:]}" if len(year) == 8 else year
    emb = metas["embeddings"]
    umap = metas["umap"]
    raw = metas.get("raw", {})
    paragraphs = [
        (
            f"Every course in Stanford University's {year_text} catalog (ExploreCourses, fetched {fetched}) placed on "
            f"a two-dimensional map by the meaning of its title and description. This deposit is the auditable record "
            f"behind the interactive map at {config.MAP_URL}: the cleaned corpus of {counts.get('kept', 0):,} courses "
            f"(from {metas['corpus'].get('n_in', 0):,} in the catalog, after preregistered rules that drop "
            f"{counts.get('rule_1_placeholder_dropped', 0)} placeholder descriptions and "
            f"{counts.get('rule_2_administrative_dropped', 0)} cross-departmental administrative boilerplate "
            f"descriptions and strip programme template text from {counts.get('rule_3_template_stripped', 0)} more), "
            f"the embeddings from every candidate model in the preregistered model comparison, the UMAP layout, the "
            f"Toponymy regions ({regions} at four granularities, finest first) with their names, and the evaluation "
            f"record: layout metrics, region stability over UMAP seeds and subsamples, two name evaluations, and "
            f"validation checks against findings in the literature."
        ),
        (
            f"Position comes from {emb.get('model', 'the embedding model')} (revision {emb.get('revision', '?')}, "
            f"{emb.get('dimension', '?')} dimensions, unit-normalised) applied to title and description only, reduced "
            f"with UMAP (n_neighbors {umap.get('n_neighbors', '?')}, min_dist {umap.get('min_dist', '?')}, "
            f"{umap.get('metric', '?')} metric, random_state {umap.get('random_state', '?')}). Regions are density "
            f"clusters in the 2-d layout found by Toponymy and named by a large language model "
            f"({metas['labels'].get('namer_model', '?')}); the names are LLM-generated labels for browsing, not "
            f"ground truth, and the record's name evaluations measure how well they identify their regions."
        ),
        (
            "Files: corpus.parquet (the cleaned corpus with the embedded text), courses.parquet (the catalog before "
            "the corpus rules, one row per registrar course with every cross-listed code), labels.parquet, "
            "umap_coords.npz, topic_names.json and cluster_tree.json (the published regions, layout and names), "
            "embeddings*.npz (one "
            "per candidate model, on the cleaned corpus and on the raw catalog), record_<source>.tar.gz (the versioned "
            "record directory: corpus rules and drops, run metadata, the preregistered comparison, layout metrics, "
            "stability, the name evaluations with every listener call, the validation checks), "
            "exploration_<source>.tar.gz (the two raw-catalog exploration maps, the other candidates' layouts, the "
            "EVoC layers of the post-hoc structure analysis), raw_checksums.json, and the code at the archived commit."
        ),
        (
            f"Not included: the raw ExploreCourses XML ({raw.get('n_files', '?')} files, "
            f"{raw.get('total_bytes', 0) / 1e6:.0f} MB). {TERMS['how_read']} Stanford's terms state: "
            f'"{TERMS["terms_quote_download"][:-1]}" and "{TERMS["terms_quote_copy"][:-1]}"; the About page '
            f'describes the course data API as "available to students and faculty". {TERMS["decision"]}'
        ),
        (
            f"Code and methodology: {config.REPO_URL} at commit {commit[:12]}; the preregistration, the comparison "
            f"result and the completion plan are in its docs/ directory. The derived data here are released under "
            f"CC BY 4.0. Course titles and descriptions are Stanford University's."
            + (f" Cite as DOI {doi}." if doi else "")
        ),
    ]
    return paragraphs


def description_html(paragraphs: list[str]) -> str:
    return "\n".join(f"<p>{p}</p>" for p in paragraphs)


def build_metadata(source: str, metas: dict, commit: str, doi: str | None = None) -> dict:
    creator = dict(config.ARCHIVE_CREATOR)
    if config.ARCHIVE_ORCID:
        creator["orcid"] = config.ARCHIVE_ORCID
    year = metas["manifest"].get("academic_year", "")
    year_text = f"{year[:4]}-{year[6:]}" if len(year) == 8 else year
    return {
        "upload_type": "dataset",
        "publication_date": date.today().isoformat(),
        "title": (
            f"{source.capitalize()} course catalog map ({year_text}): corpus, embeddings, layout, region names, "
            f"and evaluation record"
        ),
        "creators": [creator],
        "description": description_html(describe(source, metas, commit, doi)),
        "access_right": "open",
        "license": config.ARCHIVE_LICENSE,
        "keywords": KEYWORDS,
        "version": config.ARCHIVE_VERSION,
        "language": "eng",
        "related_identifiers": [
            {"identifier": config.REPO_URL, "relation": "isSupplementTo", "resource_type": "software"},
            {"identifier": config.MAP_URL, "relation": "isSupplementTo"},
            {"identifier": config.STANFORD_BASE_URL, "relation": "isDerivedFrom", "resource_type": "dataset"},
        ],
        "prereserve_doi": True,
    }


def readme_markdown(source: str, metas: dict, commit: str, doi: str | None, files: dict) -> str:
    lines = [f"# {source.capitalize()} course catalog map: archive", ""]
    lines += [p + "\n" for p in describe(source, metas, commit, doi)]
    lines += ["## Files", "", "| file | bytes | sha256 |", "|---|---:|---|"]
    for name in sorted(n for n in files if n != "README.md"):
        lines.append(f"| {name} | {files[name]['bytes']:,} | {files[name]['sha256']} |")
    lines += ["", f"Archived commit: {commit}. Zenodo lists this README's own checksum.", ""]
    return "\n".join(lines)


# ── State (archive.json) ──────────────────────────────────────────────────────
def state_path(paths: dict) -> Path:
    return paths["base"] / "archive.json"


def load_state(paths: dict, sandbox: bool) -> dict:
    p = state_path(paths)
    if p.exists():
        return json.loads(p.read_text())
    return {
        "source": paths["source"],
        "version": config.ARCHIVE_VERSION,
        "zenodo": {
            "host": "sandbox.zenodo.org" if sandbox else "zenodo.org",
            "sandbox": sandbox,
            "deposition_id": None,
            "doi": None,
            "doi_url": None,
            "concept_doi": None,
            "record_url": None,
            "reserved_at": None,
            "published_at": None,
        },
        "commit": {},
        "terms": TERMS,
        "raw_snapshot": {"included": False},
        "files": {},
    }


def save_state(paths: dict, state: dict) -> None:
    atomic_write_text(state_path(paths), json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Zenodo ───────────────────────────────────────────────────────────────────
class Zenodo:
    """The legacy deposit API (create, bucket upload, publish), which Zenodo keeps supporting on InvenioRDM."""

    def __init__(self, sandbox: bool, token: str, dry_run: bool):
        self.base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.dry_run = dry_run
        self.token_var = "ZENODO_SANDBOX_TOKEN" if sandbox else "ZENODO_TOKEN"
        self.token = token

    def _require_token(self) -> None:
        if not self.token and not self.dry_run:
            sys.exit(f"{self.token_var} is not set (export it in ~/.secrets or put it in .env)")

    def _check(self, r: requests.Response) -> dict:
        if r.status_code >= 400:
            sys.exit(f"Zenodo {r.request.method} {r.url} -> {r.status_code}: {r.text[:2000]}")
        return r.json() if r.text else {}

    def create(self, metadata: dict) -> dict | None:
        self._require_token()
        if self.dry_run:
            print("DRY RUN: POST /deposit/depositions with metadata:")
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
            return None
        return self._check(
            requests.post(f"{self.base}/deposit/depositions", json={"metadata": metadata}, headers=self.headers)
        )

    def get(self, dep_id: int) -> dict:
        self._require_token()
        return self._check(requests.get(f"{self.base}/deposit/depositions/{dep_id}", headers=self.headers))

    def update_metadata(self, dep_id: int, metadata: dict) -> dict | None:
        self._require_token()
        if self.dry_run:
            print(f"DRY RUN: PUT /deposit/depositions/{dep_id} with metadata:")
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
            return None
        return self._check(
            requests.put(f"{self.base}/deposit/depositions/{dep_id}", json={"metadata": metadata}, headers=self.headers)
        )

    def upload(self, bucket_url: str, path: Path) -> dict | None:
        self._require_token()
        if self.dry_run:
            print(f"DRY RUN: PUT {bucket_url}/{path.name} ({path.stat().st_size:,} bytes)")
            return None
        with open(path, "rb") as f:
            return self._check(requests.put(f"{bucket_url}/{path.name}", data=f, headers=self.headers))

    def publish(self, dep_id: int) -> dict | None:
        self._require_token()
        if self.dry_run:
            print(f"DRY RUN: POST /deposit/depositions/{dep_id}/actions/publish")
            return None
        return self._check(
            requests.post(f"{self.base}/deposit/depositions/{dep_id}/actions/publish", headers=self.headers)
        )


# ── Actions ──────────────────────────────────────────────────────────────────
def action_checksums(paths: dict, state: dict) -> None:
    base = paths["base"]
    raw = raw_checksums(paths["raw"])
    out = base / "raw_checksums.json"
    atomic_write_text(out, json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    print(f"raw snapshot: {raw['n_files']} files, {raw['total_bytes'] / 1e6:.1f} MB, checksums -> {out.name}")
    state["raw_snapshot"] = {
        "included": False,
        "n_files": raw["n_files"],
        "total_bytes": raw["total_bytes"],
        "fetched_at": (raw["manifest"] or {}).get("fetched_at"),
        "raw_checksums_sha256": digest(out),
        "reason": TERMS["decision"],
    }
    files = deposit_data_files(base)
    for p in files:
        entry = state["files"].setdefault(p.name, {})
        entry.update({"bytes": p.stat().st_size, "sha256": digest(p), "kind": "data"})
        print(f"  {p.name:40s} {p.stat().st_size / 1e6:8.1f} MB  {entry['sha256'][:16]}")
    print(f"{len(files)} data files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    print(f"exploration bundle will hold {len(exploration_files(base))} files")


def action_reserve(paths: dict, state: dict, zen: Zenodo) -> None:
    metas = load_metas(paths["base"])
    commit = git_commit()
    metadata = build_metadata(paths["source"], metas, commit, state["zenodo"]["doi"])
    dep_id = state["zenodo"]["deposition_id"]
    if dep_id:
        print(f"deposition {dep_id} exists on {state['zenodo']['host']}; updating its metadata")
        dep = zen.update_metadata(dep_id, metadata)
    else:
        dep = zen.create(metadata)
    if dep is None:  # dry run
        return
    doi = dep["metadata"]["prereserve_doi"]["doi"]
    state["zenodo"].update(
        {
            "deposition_id": dep["id"],
            "doi": doi,
            "doi_url": f"https://doi.org/{doi}",
            "bucket_url": dep["links"]["bucket"],
            "reserved_at": state["zenodo"].get("reserved_at") or now(),
        }
    )
    state["commit"]["reserved"] = commit
    print(f"deposition {dep['id']} on {state['zenodo']['host']}: reserved DOI {doi}")
    print("next: run stage 07, put the DOI in README.md, commit, then --upload")


def action_upload(paths: dict, state: dict, zen: Zenodo) -> None:
    base, source = paths["base"], paths["source"]
    if not state["zenodo"].get("deposition_id") and not zen.dry_run:
        sys.exit("no deposition yet: run --reserve first")
    if git_dirty():
        print("WARNING: the working tree is dirty; the source zip comes from HEAD, the record tarball from disk")
    commit = git_commit()
    bundle_dir = base / "archive"
    bundle_dir.mkdir(exist_ok=True)
    record_dir = config.DOCS_DIR / "record" / source
    record_tar = bundle_dir / f"record_{source}.tar.gz"
    make_tarball(record_tar, sorted(p for p in record_dir.iterdir() if p.is_file()), f"record_{source}")
    exploration_tar = bundle_dir / f"exploration_{source}.tar.gz"
    make_tarball(exploration_tar, exploration_files(base), f"exploration_{source}")
    source_zip = bundle_dir / f"course-catalog-map-{commit[:12]}.zip"
    git_archive(source_zip, f"course-catalog-map-{commit[:12]}")
    for p in (record_tar, exploration_tar, source_zip):
        state["files"][p.name] = {"bytes": p.stat().st_size, "sha256": digest(p), "kind": "bundle"}
    # Drop stale bundle entries (an earlier commit's source zip) so the README lists only what is uploaded.
    for name in list(state["files"]):
        if state["files"][name].get("kind") == "bundle" and not (bundle_dir / name).exists():
            del state["files"][name]
    metas = load_metas(base)
    readme = bundle_dir / "README.md"
    atomic_write_text(readme, readme_markdown(source, metas, commit, state["zenodo"]["doi"], state["files"]))
    state["files"]["README.md"] = {"bytes": readme.stat().st_size, "sha256": digest(readme), "kind": "readme"}
    state["commit"]["uploaded"] = commit

    to_upload = [
        (name, base / name if entry["kind"] == "data" else bundle_dir / name) for name, entry in state["files"].items()
    ]
    for name, path in to_upload:
        if not path.exists():
            sys.exit(f"{path} is missing; run --checksums again")
    total = sum(p.stat().st_size for _, p in to_upload)
    print(f"{len(to_upload)} files, {total / 1e6:.1f} MB to upload")
    if zen.dry_run:
        for name, p in to_upload:
            print(f"  {name:48s} {p.stat().st_size / 1e6:8.1f} MB")
        return
    dep = zen.get(state["zenodo"]["deposition_id"])
    bucket = dep["links"]["bucket"]
    present = {f["filename"]: f["checksum"].removeprefix("md5:") for f in dep.get("files", [])}
    for name, p in to_upload:
        local_md5 = digest(p, "md5")
        if present.get(name) == local_md5:
            print(f"  {name}: already uploaded (md5 matches)")
            state["files"][name].update({"md5": local_md5, "uploaded": True})
            continue
        print(f"  uploading {name} ({p.stat().st_size / 1e6:.1f} MB)...", flush=True)
        r = zen.upload(bucket, p)
        remote_md5 = r["checksum"].removeprefix("md5:")
        if remote_md5 != local_md5:
            sys.exit(f"{name}: Zenodo reports md5 {remote_md5}, local is {local_md5}")
        state["files"][name].update({"md5": local_md5, "uploaded": True, "uploaded_at": now()})
        save_state(paths, state)
    print("all files uploaded and verified; next: --publish")


def action_publish(paths: dict, state: dict, zen: Zenodo) -> None:
    dep_id = state["zenodo"].get("deposition_id")
    if not dep_id and not zen.dry_run:
        sys.exit("no deposition yet: run --reserve, then --upload")
    missing = [n for n, e in state["files"].items() if not e.get("uploaded")]
    if missing and not zen.dry_run:
        sys.exit(f"not uploaded yet: {missing}")
    metas = load_metas(paths["base"])
    zen.update_metadata(dep_id, build_metadata(paths["source"], metas, git_commit(), state["zenodo"]["doi"]))
    dep = zen.publish(dep_id)
    if dep is None:
        return
    state["zenodo"].update(
        {
            "doi": dep["doi"],
            "doi_url": dep["doi_url"],
            "concept_doi": dep.get("conceptdoi"),
            "record_url": dep["links"].get("record_html") or dep["links"].get("html"),
            "published_at": now(),
        }
    )
    print(f"published: {dep['doi_url']} (concept DOI {dep.get('conceptdoi')}); run stage 07 and commit the record")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--checksums", action="store_true")
    ap.add_argument("--reserve", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--sandbox", action="store_true", help="sandbox.zenodo.org with ZENODO_SANDBOX_TOKEN")
    ap.add_argument("--dry-run", action="store_true", help="print the requests instead of sending them")
    args = ap.parse_args()
    if not (args.checksums or args.reserve or args.upload or args.publish):
        ap.error("pick at least one of --checksums, --reserve, --upload, --publish")
    paths = config.source_paths(args.source)
    state = load_state(paths, args.sandbox)
    if state["zenodo"]["sandbox"] != args.sandbox and state["zenodo"].get("deposition_id"):
        sys.exit(f"archive.json belongs to {state['zenodo']['host']}; delete it to start over on the other host")
    token = config.ZENODO_SANDBOX_TOKEN if args.sandbox else config.ZENODO_TOKEN
    zen = Zenodo(args.sandbox, token, args.dry_run)
    if args.checksums:
        action_checksums(paths, state)
        save_state(paths, state)
    if args.reserve:
        action_reserve(paths, state, zen)
        save_state(paths, state)
    if args.upload:
        action_upload(paths, state, zen)
        save_state(paths, state)
    if args.publish:
        action_publish(paths, state, zen)
        save_state(paths, state)


if __name__ == "__main__":
    main()
