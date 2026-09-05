"""Post-hoc structure analysis, step 3: name two layers of each model's EVoC hierarchy with Toponymy, so the
cross-model crosstabs carry names instead of cluster ids.

    uv run python experiments/name_evoc.py --embedding qwen3-4b --device runpod [--keep-pod] [--dry-run]

Toponymy is handed a pre-fitted (duck-typed) clusterer built from the saved EVoC layers, so it names without
re-clustering: the same pattern as jeopardy-map's analysis/10_evoc_toponymy.py. The two layers named are the
ones nearest 40 and 120 regions, the levels used in structure_numbers.py. Same namer, style note and
embedder as the map, so names are comparable across models. Not part of the preregistered comparison.
"""

import argparse
import datetime as dt
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import config  # noqa: E402
from io_utils import atomic_write_text, write_parquet_safely  # noqa: E402

TARGET_REGIONS = [120, 40]  # finest first, as Toponymy expects


class PrefitClusterer:
    """Duck-typed stand-in that Toponymy.fit accepts as an already-fitted clusterer."""

    def __init__(self, cluster_layers, cluster_tree):
        self.cluster_layers_ = cluster_layers
        self.cluster_tree_ = cluster_tree


def evoc_file(files: dict, name: str, ext: str) -> Path:
    return files["npz"].with_name(files["npz"].name.replace("embeddings", name).replace(".npz", ext))


def load_layers(files: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    d = np.load(evoc_file(files, "evoc_layers", ".npz"), allow_pickle=True)
    keys = sorted((k for k in d.files if k.startswith("layer_")), key=lambda k: int(k.split("_")[1]))
    return d["course_id"], [d[k] for k in keys]


def pick_layers(layers: list[np.ndarray]) -> list[np.ndarray]:
    chosen = []
    for target in TARGET_REGIONS:
        lab = min(layers, key=lambda x: abs(x.max() + 1 - target))
        if not any(lab is c for c in chosen):
            chosen.append(lab)
    return chosen


def name_locally(key: str, device: str | None, dry_run: bool) -> None:
    from toponymy import Toponymy
    from toponymy.cluster_layer import ClusterLayerText
    from toponymy.clustering import build_cluster_tree, centroids_from_labels

    from embedder import CourseEmbedder
    from sources import get_source

    stage04 = importlib.import_module("04_label_topics")
    paths = config.source_paths(config.DEFAULT_SOURCE)
    files = config.keyed_files(paths, key)
    corpus = pd.read_parquet(paths["corpus"], columns=["course_id", "embed_text"])
    emb = np.load(files["npz"], allow_pickle=True)["embeddings"]
    ids, all_layers = load_layers(files)
    assert (ids == corpus["course_id"].to_numpy()).all()
    label_layers = pick_layers(all_layers)
    counts = [int(lab.max()) + 1 for lab in label_layers]
    print(f"{key}: naming EVoC layers with {counts} regions ({sum(counts)} names)", flush=True)
    if dry_run:
        return

    layers = [ClusterLayerText(lab, centroids_from_labels(lab, emb), layer_id=i) for i, lab in enumerate(label_layers)]
    clusterer = PrefitClusterer(layers, build_cluster_tree(label_layers))
    source = get_source(config.DEFAULT_SOURCE)
    topic_model = Toponymy(
        llm_wrapper=stage04.make_namer(),
        text_embedding_model=CourseEmbedder(key=key, device=device),
        clusterer=clusterer,
        object_description=source.OBJECT_DESCRIPTION,
        corpus_description=source.CORPUS_DESCRIPTION,
        verbose=True,
    )
    t0 = time.time()
    topic_model.fit(objects=corpus["embed_text"].tolist(), embedding_vectors=emb, clusterable_vectors=emb)
    minutes = (time.time() - t0) / 60

    out = pd.DataFrame({"course_id": ids})
    for i, (lab, names) in enumerate(zip(label_layers, topic_model.topic_names_)):
        out[f"evoc_cluster_{i}"] = lab.astype(np.int32)
        out[f"evoc_label_{i}"] = np.where(
            lab >= 0, np.asarray(names, dtype=object)[np.clip(lab, 0, None)], "Unlabelled"
        )
    write_parquet_safely(out, evoc_file(files, "evoc_labels", ".parquet"))
    atomic_write_text(
        evoc_file(files, "evoc_topic_names", ".json"),
        json.dumps({f"layer_{i}": list(map(str, n)) for i, n in enumerate(topic_model.topic_names_)}, indent=1),
    )
    meta = {
        "embedding": key,
        "layers": counts,
        "namer_model": config.NAMER_MODEL,
        "namer_style": config.NAMER_STYLE,
        "minutes": round(minutes, 1),
        "named_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(evoc_file(files, "evoc_labels_meta", ".json"), json.dumps(meta, indent=2))
    print(f"{key}: named in {minutes:.1f} min; coarse layer: {list(topic_model.topic_names_[-1])}")


def name_on_runpod(key: str, keep_pod: bool) -> None:
    from remote import run_on_pod

    paths = config.source_paths(config.DEFAULT_SOURCE)
    files = config.keyed_files(paths, key)
    rel = lambda p: str(Path(p).relative_to(config.ROOT))  # noqa: E731
    outputs = [
        evoc_file(files, n, e)
        for n, e in (("evoc_labels", ".parquet"), ("evoc_topic_names", ".json"), ("evoc_labels_meta", ".json"))
    ]
    run_on_pod(
        uploads=[
            "pipeline",
            "experiments",
            rel(paths["corpus"]),
            rel(files["npz"]),
            rel(evoc_file(files, "evoc_layers", ".npz")),
        ],
        command=(
            f"cd {config.RUNPOD_REMOTE_DIR} && env HF_HOME={config.RUNPOD_HF_HOME} PYTHONUNBUFFERED=1 "
            f"TOKENIZERS_PARALLELISM=false python experiments/name_evoc.py --embedding {key} --device cuda"
        ),
        downloads=[(f"{config.RUNPOD_REMOTE_DIR}/{rel(f)}", f) for f in outputs],
        env={"ANTHROPIC_API_KEY": config.ANTHROPIC_API_KEY},
        extra_packages=config.RUNPOD_LABEL_PACKAGES,
        keep_pod=keep_pod,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu | runpod")
    ap.add_argument("--keep-pod", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report which layers would be named; no LLM calls")
    args = ap.parse_args()
    if args.device == "runpod":
        name_on_runpod(args.embedding, args.keep_pod)
        return
    name_locally(args.embedding, None if args.device == "auto" else args.device, args.dry_run)


if __name__ == "__main__":
    main()
