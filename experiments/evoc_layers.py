"""Post-hoc structure analysis, step 1: cluster each model's embeddings directly with EVoC (no 2-d layout in
the way) and save the layer hierarchy -> data/<source>/evoc_layers[_<key>].npz.

    uv run python experiments/evoc_layers.py --models qwen3-0.6b qwen3-4b arctic-l-v2 nomic-v2-moe

Not part of the preregistered comparison; the model choice is closed. EVoC is seeded so the layers are
reproducible on this machine.
"""

import argparse
import sys
import time
from pathlib import Path

import evoc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

BASE_MIN_CLUSTER_SIZE = 10  # matches the map's ToponymyClusterer setting
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--models", nargs="+", default=list(config.EMBED_MODELS))
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    for key in args.models:
        files = config.keyed_files(paths, key, raw=args.raw)
        data = np.load(files["npz"], allow_pickle=True)
        X = data["embeddings"]
        t0 = time.time()
        clusterer = evoc.EVoC(base_min_cluster_size=BASE_MIN_CLUSTER_SIZE, random_state=SEED)
        clusterer.fit(X)
        layers = [np.asarray(layer, dtype=np.int32) for layer in clusterer.cluster_layers_]
        # Finest first, to match Toponymy's convention.
        layers.sort(key=lambda lab: -(lab.max() + 1))
        out = files["npz"].with_name(files["npz"].name.replace("embeddings", "evoc_layers"))
        np.savez(open(out, "wb"), course_id=data["course_id"], **{f"layer_{i}": lab for i, lab in enumerate(layers)})
        summary = ", ".join(f"{lab.max() + 1} regions/{(lab < 0).mean():.0%} unlabelled" for lab in layers)
        print(f"{key}: {X.shape} -> {len(layers)} layers in {time.time() - t0:.0f}s: {summary}")
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
