"""Stage 06: the centroid-to-name cosine guardrail for a named map -> data/<source>/name_guardrail[_<key>].csv.

For every named region at every layer: embed the name with the map's own embedder and take its cosine
similarity to the region's centroid in the embedding space. Steven's Toponymy evaluation (TutteInstitute
/toponymy discussions 173) found this tracks a grounded judge at about Spearman 0.8 as a gross-failure
alarm and is a coin flip as a fine ranker, so it is used only to flag the bottom tail for a human look.
"""

import argparse

import numpy as np
import pandas as pd

import config
from embedder import CourseEmbedder

BOTTOM = 8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--embedding", default=config.EMBED_MODEL_KEY, choices=sorted(config.EMBED_MODELS))
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    files = config.keyed_files(paths, args.embedding, raw=args.raw)
    emb = np.load(files["npz"], allow_pickle=True)["embeddings"]
    labels = pd.read_parquet(files["labels"])
    embedder = CourseEmbedder(key=args.embedding)
    layer_cols = sorted(
        (c for c in labels.columns if c.startswith("cluster_layer_")), key=lambda c: int(c.rsplit("_", 1)[1])
    )

    rows = []
    for col in layer_cols:
        layer = int(col.rsplit("_", 1)[1])
        cl = labels[col].to_numpy()
        names = labels[f"label_layer_{layer}"].to_numpy()
        ids = sorted(set(cl[cl >= 0]))
        region_names = [names[cl == i][0] for i in ids]
        name_vecs = embedder.encode(region_names)
        for i, name, nv in zip(ids, region_names, name_vecs):
            members = emb[cl == i]
            centroid = members.mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            rows.append(
                {"layer": layer, "cluster": i, "name": name, "n": int(len(members)), "cosine": float(nv @ centroid)}
            )
    df = pd.DataFrame(rows)
    df["rank_in_layer"] = df.groupby("layer")["cosine"].rank(pct=True)
    out = files["labels"].with_name(
        files["labels"].name.replace("labels", "name_guardrail").replace(".parquet", ".csv")
    )
    df.sort_values(["layer", "cosine"]).to_csv(out, index=False)
    print(f"wrote {out}")
    for layer, g in df.groupby("layer"):
        med, lo = g["cosine"].median(), g["cosine"].min()
        print(f"\nlayer {layer}: {len(g)} regions, cosine median {med:.3f}, min {lo:.3f}; lowest {BOTTOM}:")
        for r in g.nsmallest(BOTTOM, "cosine").itertuples():
            print(f"  {r.cosine:.3f}  n={r.n:4d}  {r.name}")


if __name__ == "__main__":
    main()
