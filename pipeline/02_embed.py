"""Stage 02: embed each course's title + description -> data/<source>/embeddings.npz (+ embeddings_meta.json).

`--model <key>` picks an entry of config.EMBED_MODELS; keys other than the default write a suffixed
embeddings_<key>.npz so several models can be compared side by side. `--device runpod` runs this same
script on a Runpod GPU pod (see remote.py) and pulls the result back; any other device runs locally.

Resumable: chunks land in an embed_cache directory as they finish and are skipped on re-run; the
cache is removed once the final file is verified.
"""

import argparse
import datetime as dt
import json
import shutil
import time

import numpy as np
import pandas as pd

import config
from embedder import CourseEmbedder, compose_embed_text
from io_utils import atomic_write_text


def load_texts(paths: dict, raw: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Cleaned corpus with its stage-01b embed_text, or the raw catalog as title + description."""
    if raw:
        courses = pd.read_parquet(paths["courses"], columns=["course_id", "title", "description"])
        return courses, compose_embed_text(courses)
    corpus = pd.read_parquet(paths["corpus"], columns=["course_id", "embed_text"])
    return corpus, corpus["embed_text"]


def embed_locally(source: str, key: str, device: str | None, limit: int | None, raw: bool = False) -> None:
    paths = config.source_paths(source)
    files = config.keyed_files(paths, key, raw=raw)
    courses, texts = load_texts(paths, raw)
    if limit:
        courses, texts = courses.head(limit), texts.head(limit)
    n = len(texts)
    print(f"{n} courses; text length words: median {texts.str.split().str.len().median():.0f}")

    embedder = CourseEmbedder(key=key, device=device)
    print("embedder:", json.dumps(embedder.describe()))

    if limit:
        t0 = time.time()
        vecs = embedder.encode(texts.tolist(), show_progress_bar=True)
        rate = n / (time.time() - t0)
        print(f"{vecs.shape} in {n / rate:.1f}s ({rate:.1f} docs/s); full corpus estimate: {11500 / rate / 60:.1f} min")
        return

    cache = files["cache"]
    cache.mkdir(parents=True, exist_ok=True)
    n_chunks = (n + config.EMBED_CHUNK_SIZE - 1) // config.EMBED_CHUNK_SIZE
    todo = [i for i in range(n_chunks) if not (cache / f"chunk_{i:04d}.npy").exists()]
    print(f"{n_chunks} chunks of {config.EMBED_CHUNK_SIZE}: {n_chunks - len(todo)} cached, {len(todo)} to embed")
    t0 = time.time()
    for k, i in enumerate(todo, start=1):
        lo, hi = i * config.EMBED_CHUNK_SIZE, min((i + 1) * config.EMBED_CHUNK_SIZE, n)
        vecs = embedder.encode(texts.iloc[lo:hi].tolist(), show_progress_bar=False)
        assert vecs.shape[0] == hi - lo and np.isfinite(vecs).all()
        tmp = cache / f"chunk_{i:04d}.npy.tmp"
        with open(tmp, "wb") as fh:  # a file handle, or np.save appends ".npy" to the tmp name
            np.save(fh, vecs)
        tmp.rename(cache / f"chunk_{i:04d}.npy")
        rate = (k * config.EMBED_CHUNK_SIZE) / (time.time() - t0)
        print(f"[{k}/{len(todo)}] chunk {i}: {vecs.shape}, {rate:.1f} docs/s", flush=True)

    embeddings = np.concatenate([np.load(cache / f"chunk_{i:04d}.npy") for i in range(n_chunks)])
    assert embeddings.shape == (n, embeddings.shape[1]) and np.isfinite(embeddings).all()
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"embeddings not unit norm: {norms.min():.4f}..{norms.max():.4f}"

    tmp = files["npz"].with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:  # a file handle, or np.savez appends ".npz" to the tmp name
        np.savez(fh, course_id=courses["course_id"].to_numpy(), embeddings=embeddings)
    check = np.load(tmp, allow_pickle=True)
    assert check["embeddings"].shape == embeddings.shape and len(check["course_id"]) == n
    tmp.rename(files["npz"])
    meta = embedder.describe() | {
        "n_documents": int(n),
        "dimension": int(embeddings.shape[1]),
        "corpus": "raw catalog, title + newline + description"
        if raw
        else "corpus.parquet embed_text (stage 01b rules)",
        "embedded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(files["meta"], json.dumps(meta, indent=2))
    shutil.rmtree(cache)
    print(f"wrote {files['npz']} {embeddings.shape} and {files['meta'].name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--model", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--device", default="auto", help="auto | mps | cuda | cpu | runpod")
    ap.add_argument("--limit", type=int, help="embed only the first N courses (smoke test; writes nothing)")
    ap.add_argument("--keep-pod", action="store_true", help="runpod only: leave the pod running afterwards")
    ap.add_argument("--raw", action="store_true", help="embed the raw 11,500-course catalog (sensitivity row) instead")
    args = ap.parse_args()

    if args.device == "runpod":
        from remote import run_embed_on_runpod

        run_embed_on_runpod(args.source, args.model, limit=args.limit, keep_pod=args.keep_pod, raw=args.raw)
        return
    embed_locally(args.source, args.model, None if args.device == "auto" else args.device, args.limit, raw=args.raw)


if __name__ == "__main__":
    main()
