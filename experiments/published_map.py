"""The published map as one object, for the instruments that evaluate its names (steps 2 and 3 of docs/plan.md).

Loads the cleaned corpus, the embeddings, the layout, the published labels and the cluster tree, all aligned to
corpus.parquet row order, and answers the structural questions the instruments ask: who is in a region, where its
centroid is, who its parent and coarsest ancestor are.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

Region = tuple[int, int]  # (layer, cluster); layer 0 is the finest


@dataclass
class PublishedMap:
    course_id: np.ndarray
    texts: list[str]  # embed_text: the text the embedder and the namer saw
    X: np.ndarray  # unit-norm embeddings, aligned to course_id
    Y: np.ndarray  # 2-d layout, aligned to course_id
    layers: list[np.ndarray]  # cluster id per course per layer, finest first, -1 = unlabelled
    names: list[list[str]]  # region names per layer, indexed by cluster id
    parent: dict[Region, Region]  # region -> parent region; the root is (len(layers), 0)
    _centroids: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def root(self) -> Region:
        return (self.n_layers, 0)

    def n_regions(self, layer: int) -> int:
        return len(self.names[layer])

    def members(self, layer: int, cluster: int) -> np.ndarray:
        return np.flatnonzero(self.layers[layer] == cluster)

    def name(self, region: Region) -> str:
        return self.names[region[0]][region[1]]

    def centroids(self, layer: int, space: str = "embedding") -> np.ndarray:
        """Mean member vector per region (unit-normalised in the embedding space)."""
        key = (layer, space)
        if key not in self._centroids:
            V = self.X if space == "embedding" else self.Y
            labels = self.layers[layer]
            out = np.zeros((self.n_regions(layer), V.shape[1]))
            for c in range(self.n_regions(layer)):
                out[c] = V[labels == c].mean(axis=0)
            if space == "embedding":
                out /= np.linalg.norm(out, axis=1, keepdims=True)
            self._centroids[key] = out
        return self._centroids[key]

    def children(self, region: Region) -> list[Region]:
        return [r for r, p in self.parent.items() if p == region]

    def descendants(self, region: Region, layer: int) -> list[int]:
        """Cluster ids at `layer` below `region`, through any number of intermediate layers."""
        out: list[int] = []
        for kid in self.children(region):
            if kid[0] == layer:
                out.append(kid[1])
            elif kid[0] > layer:
                out += self.descendants(kid, layer)
        return sorted(out)

    def coarsest_ancestor(self, region: Region) -> Region:
        """The topmost named region above `region` (itself if its parent is the root)."""
        while self.parent[region] != self.root:
            region = self.parent[region]
        return region


def parse_tree(tree: dict[str, list[str]]) -> dict[Region, Region]:
    """cluster_tree.json (parent "L:c" -> children ["l:c", ...]) as a child -> parent map."""
    parent: dict[Region, Region] = {}
    for p, kids in tree.items():
        pl, pc = (int(x) for x in p.split(":"))
        for k in kids:
            kl, kc = (int(x) for x in k.split(":"))
            assert (kl, kc) not in parent, f"{k} has two parents"
            parent[(kl, kc)] = (pl, pc)
    return parent


def load_published(source: str = config.DEFAULT_SOURCE) -> PublishedMap:
    paths = config.source_paths(source)
    files = config.keyed_files(paths)
    corpus = pd.read_parquet(paths["corpus"])
    ids = corpus["course_id"].to_numpy()
    emb = np.load(files["npz"], allow_pickle=True)
    lay = np.load(files["umap"], allow_pickle=True)
    labels = pd.read_parquet(files["labels"])
    assert (emb["course_id"] == ids).all() and (lay["course_id"] == ids).all() and (labels["course_id"] == ids).all()
    layer_cols = sorted((c for c in labels.columns if c.startswith("cluster_layer_")), key=lambda c: int(c[14:]))
    layers = [labels[c].to_numpy().astype(np.int32) for c in layer_cols]
    topic_names = json.loads(Path(files["topic_names"]).read_text())
    names = [topic_names[f"layer_{i}"] for i in range(len(layers))]
    for lab, nm in zip(layers, names):
        assert lab.max() + 1 == len(nm), "topic_names.json does not match labels.parquet"
    parent = parse_tree(json.loads(Path(files["cluster_tree"]).read_text()))
    assert all((i, c) in parent for i, nm in enumerate(names) for c in range(len(nm))), "regions missing from the tree"
    return PublishedMap(
        course_id=ids,
        texts=corpus["embed_text"].tolist(),
        X=emb["embeddings"],
        Y=lay["coords"],
        layers=layers,
        names=names,
        parent=parent,
    )
