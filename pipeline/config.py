"""Shared paths, constants, and env loading. Every pipeline script imports from here."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)  # a repo-local .env; keys otherwise arrive through the shell environment (~/.secrets)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

# ── Source (university) ──────────────────────────────────────────────────────
# Each university is an adapter in pipeline/sources/ that fetches its own raw
# catalog and normalises it to the common schema in sources/__init__.py.
# Everything downstream of stage 01 is source-agnostic.
DEFAULT_SOURCE = "stanford"


def source_paths(source: str) -> dict[str, Path]:
    base = DATA_DIR / source
    return {
        "source": source,
        "base": base,
        "raw": base / "raw",
        "listings": base / "listings.parquet",  # one row per (course, code) listing, pre-dedupe
        "courses": base / "courses.parquet",  # one row per course, common schema
        "drops": base / "drops.csv",  # every row removed between listings and courses, with the rule
        "corpus": base / "corpus.parquet",  # stage 01b: courses.parquet after the preregistered rules, + embed_text
        "corpus_drops": base / "corpus_drops.csv",  # every course removed by those rules, with the rule
        "corpus_meta": base / "corpus_meta.json",
        "embeddings": base / "embeddings.npz",
        "umap": base / "umap_coords.npz",
        "labels": base / "labels.parquet",
        "toponymy_model": base / "toponymy_model.joblib",
        "map_html": base / f"{source}_course_map.html",
    }


# ── Corpus rules (stage 01b; docs/preregistration.md part 2) ─────────────────
CORPUS_MIN_DESCRIPTION_WORDS = 3  # rule 1: fewer words than this is a placeholder ("TBD", "TGR Dissertation")
CORPUS_SHARED_MIN_COURSES = 5  # rules 2 and 3: how many courses must share text before it counts as shared
CORPUS_SHARED_MIN_DEPARTMENTS = 5  # rule 2: shared across this many departments makes it administrative
CORPUS_TEMPLATE_MIN_WORDS = 10  # rule 3: a shared opening shorter than this is left alone

# ── HTTP ─────────────────────────────────────────────────────────────────────
USER_AGENT = "course-catalog-map/0.1 (research datamap of public course catalogs)"
REQUEST_TIMEOUT_S = 180  # department payloads run to ~15 MB because sections are inline
REQUEST_DELAY_S = 0.5

# ── Stanford ExploreCourses ──────────────────────────────────────────────────
STANFORD_BASE_URL = "https://explorecourses.stanford.edu/"
STANFORD_XML_VIEW = "xml-20200810"
STANFORD_ACADEMIC_YEAR = "20262027"  # the catalog year ExploreCourses served by default on 2026-09-04

# ── API keys ─────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── Embedding models (stage 02; the same model embeds Toponymy's keyphrases in stage 04) ──
# Qwen3-Embedding's instruction format. The technical report evaluates MTEB clustering tasks with
# "Identify the topic or theme of the given ..." instructions, so this is the intended usage for a
# topic map rather than a retrieval one. The same instruction is applied to every text we embed.
QWEN3_INSTRUCTION = "Instruct: Identify the topic or theme of the given university course description\nQuery: "

# Every model is pinned to a Hugging Face commit. `prompt` is prepended to every text (documents and
# Toponymy's keyphrases alike); an empty prompt means the model's plain document mode.
EMBED_MODELS = {
    "qwen3-0.6b": {
        "model": "Qwen/Qwen3-Embedding-0.6B",  # Apache-2.0; Qwen3 Embedding technical report, 2025
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",  # resolved 2026-09-05
        "prompt": QWEN3_INSTRUCTION,
        "max_seq_length": 1024,  # p90 description is ~210 words; the longest run to a few hundred more
        "trust_remote_code": False,
        "family": "Qwen3 decoder-only LLM, last-token pooling",
    },
    "qwen3-4b": {
        "model": "Qwen/Qwen3-Embedding-4B",  # Apache-2.0; same family, larger
        "revision": "5cf2132abc99cad020ac570b19d031efec650f2b",  # resolved 2026-09-05
        "prompt": QWEN3_INSTRUCTION,
        "max_seq_length": 1024,
        "batch_size": 4,  # 16 GB of fp32 weights on a 24 GB card leaves little room for a 32 x 1024-token batch
        "trust_remote_code": False,
        "family": "Qwen3 decoder-only LLM, last-token pooling",
    },
    "arctic-l-v2": {
        "model": "Snowflake/snowflake-arctic-embed-l-v2.0",  # Apache-2.0; encoder family
        "revision": "ac6544c8a46e00af67e330e85a9028c66b8cfd9a",  # resolved 2026-09-05
        "prompt": "",  # document mode; the model only defines a "query: " prefix for retrieval queries
        "max_seq_length": 1024,
        "trust_remote_code": False,
        "family": "XLM-RoBERTa-large encoder, CLS pooling",
    },
    "nomic-v2-moe": {
        "model": "nomic-ai/nomic-embed-text-v2-moe",  # Apache-2.0; encoder family with an explicit clustering prefix
        "revision": "1066b6599d099fbb93dfcb64f9c37a7c9e503e85",  # resolved 2026-09-05
        "prompt": "clustering: ",
        "max_seq_length": 512,  # model maximum; a few long descriptions get truncated
        "trust_remote_code": True,  # NomicBertModel ships as remote code on the Hub
        "family": "NomicBERT mixture-of-experts encoder, mean pooling",
    },
}
EMBED_MODEL_KEY = "qwen3-0.6b"  # the map's model; other keys write embeddings_<key>.npz for comparison
EMBED_BATCH_SIZE = 32
EMBED_CHUNK_SIZE = 1_000  # checkpoint granularity for stage 02


def keyed_files(paths: dict, key: str = EMBED_MODEL_KEY, raw: bool = False) -> dict[str, Path]:
    """Every artifact downstream of stage 01b depends on which embedding model produced it and on which text.

    The map's model (EMBED_MODEL_KEY) on the cleaned corpus owns the unsuffixed names; any other key gets
    `_<key>` suffixed copies, and `raw=True` (the 11,500-course corpus before the part-2 rules, embedded as
    title + description) adds `_raw`, so exploration builds and the sensitivity row coexist with the real map.
    """
    suffix = ("_raw" if raw else "") + ("" if key == EMBED_MODEL_KEY else f"_{key}")
    base = paths["base"]
    return {
        "npz": base / f"embeddings{suffix}.npz",
        "meta": base / f"embeddings_meta{suffix}.json",
        "cache": base / f"embed_cache{suffix}",
        "umap": base / f"umap_coords{suffix}.npz",
        "umap_meta": base / f"umap_meta{suffix}.json",
        "labels": base / f"labels{suffix}.parquet",
        "topic_names": base / f"topic_names{suffix}.json",
        "cluster_tree": base / f"cluster_tree{suffix}.json",
        "labels_meta": base / f"labels_meta{suffix}.json",
        "map_html": base / f"{paths['source']}_course_map{suffix}.html",
    }


# ── Runpod (stage 02 --device runpod) ────────────────────────────────────────
RUNPOD_POD_NAME = "course-catalog-embed"
RUNPOD_TEMPLATE_ID = "runpod-torch-v280"  # runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, torch in system python
RUNPOD_GPU_ID = "NVIDIA GeForce RTX 4090"  # 24 GB: the 4B model in fp16 is ~8 GB
# Tried in order, each in the data center that currently reports the most stock. A pod created without a
# data-center constraint can be "rented" in a site with no machine and never get scheduled (seen 2026-09-05).
RUNPOD_GPU_CANDIDATES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
    "NVIDIA A40",
    "NVIDIA L4",
]
RUNPOD_CLOUD_TYPE = "SECURE"
RUNPOD_CONTAINER_DISK_GB = 40
RUNPOD_REMOTE_DIR = "/workspace/course-catalog-map"
RUNPOD_HF_HOME = "/workspace/hf-cache"
# Installed into the template's system python, which already has torch + CUDA. Pinned to the local versions.
RUNPOD_PIP_PACKAGES = [
    "sentence-transformers==5.7.0",
    "transformers==4.57.6",
    "pandas>=2.2",
    "pyarrow>=16",
    "python-dotenv>=1.0",
    "einops",  # NomicBERT remote code
]
# Stage 04 on a pod additionally needs Toponymy and its LLM client, pinned to the local versions.
RUNPOD_LABEL_PACKAGES = ["toponymy==0.5.4", "litellm==1.99.0"]

# ── Layout (stage 03) ────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS = 15
# min_dist is a clustering parameter here, not a cosmetic one: Toponymy clusters in this 2-d space,
# so regions need to come out dense enough for density clustering to find them. Keep it low.
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42  # fixed seed: forces UMAP to a single thread, worth it for a stable layout

# ── Region naming (stage 04) ─────────────────────────────────────────────────
TOPONYMY_MIN_CLUSTERS = 6
TOPONYMY_BASE_MIN_CLUSTER_SIZE = 10
NAMER_MODEL = "claude-opus-5"
NAMER_CONCURRENCY = 8
# Opus 5 rejects the temperature parameter Toponymy always sends; litellm's drop_params removes it,
# so naming runs at the model default (temperature 1.0). Names are stored, not regenerated, per run.
NAMER_PROVIDER_KWARGS = {"drop_params": True}
# Appended to Toponymy's naming prompts. Without it Opus 5 returned finest-layer names of ~11 words
# ("Sequential Russian Language Instruction: Proficiency in Speaking, Reading, Writing, and Culture"),
# which DataMapPlot wraps to six lines. Style only; it does not change which courses form a region.
NAMER_STYLE = (
    "Keep every topic name to five words or fewer, in title case, with no colon, subtitle, or list of examples."
)

# ── Rendering (stage 05) ─────────────────────────────────────────────────────
# A handful of far-flung points (placeholder and overseas-programme listings) otherwise squeeze the
# cloud into the middle third of the initial view. This trims the outermost points from the framing only.
MAP_INITIAL_ZOOM_FRACTION = 0.98


# ── Archive (stage 08; docs/plan.md step 5) ───────────────────────────────────
ZENODO_TOKEN = os.environ.get("ZENODO_TOKEN", "")  # personal token with deposit:write and deposit:actions
ZENODO_SANDBOX_TOKEN = os.environ.get("ZENODO_SANDBOX_TOKEN", "")  # sandbox.zenodo.org has separate accounts
ARCHIVE_CREATOR = {"name": "Fazzio, Steven"}
ARCHIVE_ORCID = os.environ.get("ARCHIVE_ORCID", "")  # added to the creator record when set
ARCHIVE_LICENSE = "cc-by-4.0"  # the derived data; the code carries its own licence
ARCHIVE_VERSION = "1.1"  # the completion plan's version; a rebuilt record would be a new Zenodo version
REPO_URL = "https://github.com/stevenfazzio/course-catalog-map"
MAP_URL = "https://stevenfazzio.com/course-catalog-map/"
