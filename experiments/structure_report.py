"""Post-hoc structure analysis, step 4: one interactive page characterising how the four embedding models
differ in global structure -> docs/embedding_structure.html.

    uv run python experiments/structure_report.py

Inputs: clean-corpus embeddings for every model, their EVoC layers (evoc_layers.py) and Toponymy names for two
of those layers (name_evoc.py, falls back to cluster ids when absent), the final map's labels, and the
comparison JSON. Exploratory and post-hoc: the model choice was made by docs/preregistration.md and is closed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import skew, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
import structure_numbers as sn  # noqa: E402
from name_evoc import evoc_file, pick_layers  # noqa: E402

import compare_embeddings as ce  # noqa: E402
import config  # noqa: E402

MODELS = ["qwen3-0.6b", "qwen3-4b", "arctic-l-v2", "nomic-v2-moe"]
INCUMBENT = "qwen3-0.6b"
K = 15
SANKEY_PAIRS = [("qwen3-0.6b", "arctic-l-v2"), ("qwen3-0.6b", "nomic-v2-moe"), ("qwen3-0.6b", "qwen3-4b")]
SAMPLE_FOR_GEOMETRY = 3000


def load_all():
    paths = config.source_paths(config.DEFAULT_SOURCE)
    corpus = pd.read_parquet(paths["corpus"])
    emb, evoc_named = {}, {}
    for k in MODELS:
        files = config.keyed_files(paths, k)
        d = np.load(files["npz"], allow_pickle=True)
        assert (d["course_id"] == corpus["course_id"].to_numpy()).all()
        emb[k] = d["embeddings"]
        lp = evoc_file(files, "evoc_labels", ".parquet")
        if lp.exists():
            evoc_named[k] = pd.read_parquet(lp)
    final_labels = pd.read_parquet(paths["labels"])
    comparison = json.load(open(paths["base"] / "embedding_comparison.json"))
    return paths, corpus, emb, evoc_named, final_labels, comparison


def coarse_named_layer(k, evoc_named, paths):
    """(cluster ids, names) for the ~40-region EVoC layer; names fall back to ids if the layer is unnamed."""
    if k in evoc_named:
        df = evoc_named[k]
        return df["evoc_cluster_1"].to_numpy(), df["evoc_label_1"].to_numpy()
    layers = sn.load_evoc(paths, k)
    lab = pick_layers(layers)[-1]
    return lab, np.where(lab >= 0, np.char.add("region ", lab.astype(str)), "Unlabelled")


def sankey(a_key, b_key, a, b, corpus):
    """Flow of courses between two models' named coarse EVoC regions; link hover lists top departments."""
    ca, na = a
    cb, nb = b
    both = (ca >= 0) & (cb >= 0)  # courses both models cluster; the unclustered sets would otherwise swamp the diagram
    df = pd.DataFrame({"a": na[both], "b": nb[both], "dept": corpus["department_code"].to_numpy()[both]})
    flows = (
        df.groupby(["a", "b"])
        .agg(
            n=("dept", "size"),
            depts=("dept", lambda s: ", ".join(f"{d} {c}" for d, c in s.value_counts().head(3).items())),
        )
        .reset_index()
    )
    flows = flows[flows["n"] >= 8]
    left = [x for x in pd.Series(na).value_counts().index if x in set(flows["a"])]
    right = [x for x in pd.Series(nb).value_counts().index if x in set(flows["b"])]
    nodes = [f"{a_key}: {x}" for x in left] + [f"{b_key}: {x}" for x in right]
    idx = {n: i for i, n in enumerate(nodes)}
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=[n.split(": ", 1)[1] for n in nodes],
                pad=6,
                thickness=12,
                color=["#6b7fd7"] * len(left) + ["#d77f6b"] * len(right),
            ),
            link=dict(
                source=[idx[f"{a_key}: {r.a}"] for r in flows.itertuples()],
                target=[idx[f"{b_key}: {r.b}"] for r in flows.itertuples()],
                value=flows["n"].tolist(),
                customdata=flows["depts"].tolist(),
                hovertemplate="%{source.label} → %{target.label}<br>%{value} courses<br>%{customdata}<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        title=(
            f"Coarse EVoC regions: {a_key} (left) → {b_key} (right); "
            f"{int(both.sum()):,} courses clustered by both, flows of 8+"
        ),
        height=max(700, 14 * len(nodes)),
        font_size=11,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def dissolves(a_key, b_key, a, b):
    """Regions of model A whose members model B mostly leaves unclustered."""
    ca, na = a
    cb, _ = b
    df = pd.DataFrame({"region": na, "clustered_a": ca >= 0, "unclustered_b": cb < 0})
    df = df[df["clustered_a"]]
    t = df.groupby("region").agg(n=("unclustered_b", "size"), share=("unclustered_b", "mean")).reset_index()
    t = t[t["n"] >= 20].sort_values("share", ascending=False).head(8)
    t.columns = [f"{a_key} region", "courses", f"share {b_key} leaves unclustered"]
    return t


def disagreement_by_group(emb, corpus, final_labels):
    """Per-course Jaccard of 15-NN sets between the incumbent and each other model, aggregated by facets."""
    neigh = {k: ce.knn_indices(emb[k], K) for k in MODELS}
    per = {}
    for k in MODELS:
        if k == INCUMBENT:
            continue
        inter = np.array([len(set(x) & set(y)) for x, y in zip(neigh[INCUMBENT], neigh[k])])
        per[k] = inter / (2 * K - inter)
    facets = {
        "school": corpus["school"],
        "primary format": corpus["components"].map(lambda c: (list(c) + ["none"])[0]),
        "text rule (stage 01b)": corpus["embed_text_rule"],
        "scheduled": np.where(corpus["scheduled"], "scheduled", "not scheduled"),
        "final-map region (42-layer)": final_labels["label_layer_2"],
    }
    rows = []
    for facet, values in facets.items():
        s = pd.Series(values).astype(str)
        for grp, n in s.value_counts().items():
            if n < 25:
                continue
            m = (s == grp).to_numpy()
            rows.append({"facet": facet, "group": grp, "n": int(n)} | {k: float(v[m].mean()) for k, v in per.items()})
    return pd.DataFrame(rows), per


def department_geometry(emb, corpus, min_courses=15):
    """Per model: department centroid similarity; Spearman between models; nearest department per department."""
    depts = corpus["department_code"].value_counts()
    depts = depts[depts >= min_courses].index.tolist()
    sims = {}
    for k in MODELS:
        C = np.stack([emb[k][(corpus["department_code"] == d).to_numpy()].mean(axis=0) for d in depts])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        sims[k] = C @ C.T
    iu = np.triu_indices(len(depts), 1)
    mantel = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    for a in MODELS:
        for b in MODELS:
            mantel.loc[a, b] = spearmanr(sims[a][iu], sims[b][iu]).statistic
    nearest = pd.DataFrame({"department": depts})
    for k in MODELS:
        S = sims[k].copy()
        np.fill_diagonal(S, -1)
        nearest[k] = [depts[j] for j in S.argmax(axis=1)]
    return depts, sims, mantel, nearest


def space_geometry(emb, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in MODELS:
        X = emb[k]
        idx = rng.choice(len(X), size=min(SAMPLE_FOR_GEOMETRY, len(X)), replace=False)
        S = X[idx]
        cos = S @ S.T
        mean_cos = float((cos.sum() - len(S)) / (len(S) * (len(S) - 1)))
        Xc = X - X.mean(axis=0)
        ev = np.linalg.svd(Xc[idx], compute_uv=False) ** 2
        eff_dim = float(ev.sum() ** 2 / (ev**2).sum())
        occ = np.bincount(ce.knn_indices(X, K).ravel(), minlength=len(X))
        rows.append(
            {
                "model": k,
                "dimension": X.shape[1],
                "effective dimension (participation ratio)": round(eff_dim, 1),
                "mean pairwise cosine": round(mean_cos, 3),
                "hubness (skew of 15-occurrence)": round(float(skew(occ)), 2),
                "max 15-occurrence": int(occ.max()),
            }
        )
    return pd.DataFrame(rows)


def table_html(df: pd.DataFrame, fmt="{:.3f}") -> str:
    def f(v):
        return fmt.format(v) if isinstance(v, float) else str(v)

    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{f(v)}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    paths, corpus, emb, evoc_named, final_labels, comparison = load_all()
    dept = pd.factorize(corpus["department_code"])[0]
    school = pd.factorize(corpus["school"])[0]
    evoc = {k: sn.load_evoc(paths, k) for k in MODELS}
    twod = {k: sn.layout_layers(paths, k) for k in MODELS}

    # 1. overview
    rows = []
    for k in MODELS:
        e40, t40 = sn.pick(evoc[k], 40), sn.pick(twod[k], 40)
        rows.append(
            {
                "model": k,
                "family": config.EMBED_MODELS[k]["family"].split(",")[0],
                "EVoC layers (regions)": " / ".join(str(lab.max() + 1) for lab in evoc[k]),
                "EVoC unlabelled, ~40-layer": f"{(e40 < 0).mean():.0%}",
                "AMI vs department (EVoC ~40)": sn.ami_clustered(e40, dept)[0],
                "AMI vs department (2-d ~40)": sn.ami_clustered(t40, dept)[0],
                "AMI vs school (EVoC ~40)": sn.ami_clustered(e40, school)[0],
                "AMI EVoC vs own 2-d": sn.ami_clustered(e40, t40)[0],
                "trustworthiness of 2-d": comparison["trustworthiness"][k]["mean"],
            }
        )
    overview = pd.DataFrame(rows)

    # 2. between-model AMI heatmaps
    figs = []
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("EVoC layers nearest 40 regions", "EVoC layers nearest 120 regions")
    )
    for col, target in enumerate([40, 120], start=1):
        M = np.array(
            [[sn.ami_clustered(sn.pick(evoc[a], target), sn.pick(evoc[b], target))[0] for b in MODELS] for a in MODELS]
        )
        fig.add_trace(
            go.Heatmap(
                z=M,
                x=MODELS,
                y=MODELS,
                zmin=0.6,
                zmax=1,
                colorscale="Viridis",
                text=np.round(M, 2),
                texttemplate="%{text}",
                showscale=(col == 2),
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        title="Adjusted mutual information between models' native (EVoC) partitions, on courses both cluster",
        height=420,
    )
    figs.append(("Partition agreement between models", fig))

    # 3. Sankeys with names
    dissolve_tables = []
    for a_key, b_key in SANKEY_PAIRS:
        la, lb = coarse_named_layer(a_key, evoc_named, paths), coarse_named_layer(b_key, evoc_named, paths)
        figs.append((f"Where regions go: {a_key} → {b_key}", sankey(a_key, b_key, la, lb, corpus)))
        dissolve_tables.append((f"{a_key} → {b_key}", dissolves(a_key, b_key, la, lb), dissolves(b_key, a_key, lb, la)))

    # 4. per-course disagreement by facet
    dis, per = disagreement_by_group(emb, corpus, final_labels)
    fig = go.Figure()
    order = dis[dis["facet"] != "final-map region (42-layer)"].sort_values(["facet", "n"], ascending=[True, False])
    for k in per:
        fig.add_trace(
            go.Bar(
                name=f"vs {k}",
                x=[f"{r.facet}: {r.group}" for r in order.itertuples()],
                y=order[k],
                hovertext=[f"n={r.n}" for r in order.itertuples()],
            )
        )
    fig.update_layout(
        barmode="group",
        title=f"Neighbourhood agreement with the incumbent (mean Jaccard of {K}-NN sets) by facet; higher = models agree",
        height=520,
        xaxis_tickangle=-60,
        margin=dict(b=220),
    )
    figs.append(("Which courses do the models place differently?", fig))
    lowest = (
        dis[dis["facet"] == "final-map region (42-layer)"]
        .assign(mean_agreement=lambda d: d[list(per)].mean(axis=1))
        .nsmallest(12, "mean_agreement")
    )

    # 5. department geometry
    depts, sims, mantel, nearest = department_geometry(emb, corpus)
    fam_a, fam_b = "qwen3-0.6b", "arctic-l-v2"
    differ = nearest[nearest[fam_a] != nearest[fam_b]][["department", fam_a, "qwen3-4b", fam_b, "nomic-v2-moe"]]

    # 6. space geometry
    geom = space_geometry(emb)

    # assemble
    def sec(title, body):
        return f"<section><h2>{title}</h2>{body}</section>"

    html = [
        "<title>Embedding structure</title>",
        "<style>body{font:14px/1.45 -apple-system,system-ui,sans-serif;max-width:1280px;margin:24px auto;padding:0 16px;color:#1f2328}h1{font-size:24px}h2{font-size:18px;margin-top:36px}table{border-collapse:collapse;font-size:13px;margin:8px 0 16px}th,td{border:1px solid #d0d7de;padding:4px 8px;text-align:left}th{background:#f6f8fa}.note{color:#57606a;font-size:13px}</style>",
        "<h1>How the four embedding models structure the Stanford catalog</h1>",
        f"<p class='note'>Post-hoc, exploratory. The model choice was made by <code>docs/preregistration.md</code> before any of this was computed and is closed. Cleaned corpus, {len(corpus):,} courses. EVoC clusters each model's embeddings directly (seed 42, base cluster size 10); the 2-d partitions come from the map's clusterer on each model's UMAP layout. AMI is computed on courses that both partitions cluster. Region names are one sample from Claude Opus 5 through Toponymy, with the map's five-word style note.</p>",
        sec("Overview", table_html(overview)),
    ]
    for title, fig in figs:
        html.append(sec(title, fig.to_html(full_html=False, include_plotlyjs=False)))
        if title.startswith("Where regions go"):
            keys = [f"Where regions go: {p[0]} → {p[1]}" for p in SANKEY_PAIRS]
            _, t_ab, t_ba = dissolve_tables[keys.index(title)]
            html.append(
                "<p>Regions whose members the other model mostly leaves unclustered (both directions):</p>"
                + table_html(t_ab)
                + table_html(t_ba)
            )
        if title.startswith("Which courses"):
            html.append(
                "<p>Final-map regions (42-region layer) where the other models agree least with the incumbent about neighbours:</p>"
                + table_html(lowest[["group", "n", *per, "mean_agreement"]])
            )
    html.append(
        sec(
            "Do the models agree on the large-scale arrangement of departments?",
            f"<p>Spearman correlation between models' department-centroid similarity matrices ({len(depts)} departments with 15+ courses):</p>"
            + table_html(mantel.reset_index().rename(columns={"index": "model"}))
            + f"<p>Departments whose nearest department differs between the two families ({fam_a} vs {fam_b}); {len(differ)} of {len(depts)}:</p>"
            + table_html(differ),
        )
    )
    html.append(
        sec(
            "Geometry of the spaces",
            table_html(geom, fmt="{:.3f}")
            + "<p class='note'>Effective dimension and mean cosine on a 3,000-course sample; hubness from the 15-nearest-neighbour occurrence counts over the whole corpus.</p>",
        )
    )
    out = ROOT / "docs" / "embedding_structure.html"
    page = "\n".join(html)
    page = page.replace("<h1>", '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n<h1>', 1)
    out.write_text("<!doctype html><meta charset='utf-8'>" + page)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(overview.to_string(index=False))
    print("\nMantel (Spearman of department-centroid similarities):\n", mantel.round(3).to_string())
    print(f"\n{len(differ)} of {len(depts)} departments change nearest department between families")


if __name__ == "__main__":
    main()
