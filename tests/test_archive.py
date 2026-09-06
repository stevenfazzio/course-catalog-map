"""Stage 08 (Zenodo archive): the pure parts, on synthetic files. Nothing here touches the network."""

import hashlib
import importlib
import json
import tarfile

import pytest

archive = importlib.import_module("08_archive")


@pytest.fixture
def base(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "CS.xml").write_bytes(b"<xml>cs</xml>")
    (raw / "HISTORY.xml").write_bytes(b"<xml>history</xml>" * 100)
    (raw / "_manifest.json").write_text(json.dumps({"fetched_at": "2026-09-05T03:52:18+00:00"}))
    for name in [
        "corpus.parquet",
        "courses.parquet",
        "labels.parquet",
        "umap_coords.npz",
        "topic_names.json",
        "cluster_tree.json",
        "raw_checksums.json",
        "embeddings.npz",
        "embeddings_raw.npz",
        "embeddings_arctic-l-v2.npz",
        "embeddings_meta.json",  # in the record, not the deposit
        "embeddings_meta_arctic-l-v2.json",  # exploration bundle
        "umap_coords_raw.npz",  # exploration bundle
        "umap_meta.json",  # record
        "umap_meta_qwen3-4b.json",  # exploration bundle
        "labels_raw.parquet",  # exploration bundle
        "evoc_layers.npz",  # exploration bundle
        "listings.parquet",  # nowhere
        "stanford_course_map.html",  # nowhere: rebuilt by step 6, served from GitHub
    ]:
        (tmp_path / name).write_bytes(name.encode())
    return tmp_path


def test_raw_checksums_hash_every_file_and_carry_the_manifest(base):
    out = archive.raw_checksums(base / "raw")
    assert out["n_files"] == 3
    assert [r["name"] for r in out["files"]] == ["CS.xml", "HISTORY.xml", "_manifest.json"]
    assert out["files"][0]["sha256"] == hashlib.sha256(b"<xml>cs</xml>").hexdigest()
    assert out["total_bytes"] == sum(r["bytes"] for r in out["files"])
    assert out["manifest"]["fetched_at"].startswith("2026-09-05")
    assert out["terms"]["terms_url"].startswith("https://www.stanford.edu")


def test_deposit_data_files_are_the_primaries_and_every_embedding(base):
    names = [p.name for p in archive.deposit_data_files(base)]
    assert names == sorted(
        [
            "corpus.parquet",
            "courses.parquet",
            "labels.parquet",
            "umap_coords.npz",
            "topic_names.json",
            "cluster_tree.json",
            "raw_checksums.json",
            "embeddings.npz",
            "embeddings_raw.npz",
            "embeddings_arctic-l-v2.npz",
        ]
    )


def test_exploration_files_exclude_the_maps_own_artifacts(base):
    names = {p.name for p in archive.exploration_files(base)}
    assert names == {
        "embeddings_meta_arctic-l-v2.json",
        "umap_coords_raw.npz",
        "umap_meta_qwen3-4b.json",
        "labels_raw.parquet",
        "evoc_layers.npz",
    }
    assert "umap_coords.npz" not in names and "umap_meta.json" not in names and "listings.parquet" not in names


def test_tarball_members_are_prefixed_and_sorted(base, tmp_path):
    out = tmp_path / "bundle" / "x.tar.gz"
    archive.make_tarball(out, [base / "labels_raw.parquet", base / "evoc_layers.npz"], "exploration_test")
    with tarfile.open(out) as tar:
        assert tar.getnames() == ["exploration_test/evoc_layers.npz", "exploration_test/labels_raw.parquet"]


def _metas():
    return {
        "manifest": {"academic_year": "20262027", "fetched_at": "2026-09-05T03:52:18+00:00"},
        "corpus": {
            "n_in": 11500,
            "counts": {
                "kept": 11091,
                "rule_1_placeholder_dropped": 181,
                "rule_2_administrative_dropped": 228,
                "rule_3_template_stripped": 148,
            },
        },
        "embeddings": {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614", "dimension": 1024},
        "umap": {"n_neighbors": 15, "min_dist": 0.05, "metric": "cosine", "random_state": 42},
        "labels": {"namer_model": "claude-opus-5", "layers": [{"n_clusters": n} for n in (357, 120, 42, 14)]},
        "raw": {"n_files": 256, "total_bytes": 227_400_000},
    }


def test_metadata_has_what_zenodo_requires_and_reserves_a_doi(monkeypatch):
    monkeypatch.setattr(archive.config, "ARCHIVE_ORCID", "")
    md = archive.build_metadata("stanford", _metas(), "abc123def456" + "0" * 28)
    for key in ("upload_type", "title", "creators", "description", "publication_date", "access_right", "license"):
        assert md[key]
    assert md["upload_type"] == "dataset" and md["access_right"] == "open" and md["license"] == "cc-by-4.0"
    assert md["prereserve_doi"] is True
    assert md["creators"] == [{"name": "Fazzio, Steven"}]
    assert "2026-27" in md["title"]
    assert "11,091" in md["description"] and "357 / 120 / 42 / 14" in md["description"]
    assert "not redistributed" in md["description"] and "students and faculty" in md["description"]
    assert "256 files, 227 MB" in md["description"]
    assert md["description"].startswith("<p>") and "abc123def456" in md["description"]


def test_orcid_is_attached_when_configured(monkeypatch):
    monkeypatch.setattr(archive.config, "ARCHIVE_ORCID", "0000-0002-1825-0097")
    md = archive.build_metadata("stanford", _metas(), "f" * 40, doi="10.5281/zenodo.1")
    assert md["creators"][0]["orcid"] == "0000-0002-1825-0097"
    assert "10.5281/zenodo.1" in md["description"]


def test_readme_lists_every_file_with_its_checksum():
    files = {
        "b.npz": {"bytes": 20, "sha256": "bb"},
        "a.parquet": {"bytes": 1000, "sha256": "aa"},
        "README.md": {"bytes": 1, "sha256": "self"},  # cannot list its own hash
    }
    text = archive.readme_markdown("stanford", _metas(), "c" * 40, "10.5281/zenodo.2", files)
    assert "| a.parquet | 1,000 | aa |" in text and "| b.npz | 20 | bb |" in text
    assert text.index("a.parquet") < text.index("b.npz")
    assert "| README.md |" not in text
    assert "10.5281/zenodo.2" in text


def test_dry_run_client_sends_nothing(capsys):
    zen = archive.Zenodo(sandbox=True, token="", dry_run=True)
    assert zen.create({"title": "x"}) is None
    assert zen.publish(1) is None
    out = capsys.readouterr().out
    assert "DRY RUN" in out and '"title": "x"' in out
