"""Step 2 of docs/plan.md: the wayfinding lineup, sampled over the published map's named regions.

    OMP_NUM_THREADS=1 uv run python experiments/wayfinding_lineup.py [--dry-run] [--quick] [--workers 8]

A map label's job is identification: given only the name, a reader has to find the right region among its
neighbours. The lineup measures exactly that (Steven's instrument for Toponymy, TutteInstitute/toponymy discussion
177). For a named region, a listener model sees the name and k = 5 candidate regions, the true one and its four
nearest same-layer neighbours by centroid cosine in the embedding space, each shown as five member courses (title
and description excerpt). It scores every candidate 0-100; the name's score is the probability mass on the true
region, s_true / sum(s). Chance is 0.20.

Documents are held out: Toponymy's "central" exemplar selection, which chose the eight courses the namer saw for
each region, is replayed here and those courses are excluded from the shown sample where the region has enough
other members (a name that parrots exemplar phrasing would otherwise win trivially). Distractor sets and document
samples are seeded and frozen per region, so every condition for a region faces the same lineup.

Conditions: gold (the region's own name, three samples with candidate order reshuffled, which marginalises position
bias and gives a repeat band); shuffled (the name of a random other same-layer region that is not a candidate,
the floor); gimme (the own name against four distant same-layer regions, the ceiling). Sample: forty regions at
each of the two finer layers and every region at the two coarser ones.

Writes to data/<source>/ (stage 07 copies them into the record):
    wayfinding_items.json     the frozen items: candidates, shown course ids, shuffled names, gimme distractors
    wayfinding_lineup.csv     one row per (item, condition, sample): scores in shown order, mass on true, top-1
    wayfinding_lineup.json    summary per layer and overall with bootstrap intervals, floors, ceiling, repeat band,
                              listener, prompt, usage and cost, disclosures
    wayfinding_calls.jsonl    the listener's raw answers, keyed by question id (resume cache)
"""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from toponymy.clustering import centroids_from_labels
from toponymy.exemplar_texts import diverse_exemplars

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
import config  # noqa: E402
from io_utils import atomic_write_text  # noqa: E402
from listener import Listener, have_api_key, refused  # noqa: E402
from published_map import PublishedMap, load_published  # noqa: E402

SEED = 0
K = 5  # candidates per lineup
N_DOCS = 5  # courses shown per candidate
DOC_CHARS = 500
N_EXEMPLARS = 8  # ClusterLayerText default, what stage 04 used
GOLD_SAMPLES = 3
SAMPLE_PER_LAYER = {0: 40, 1: 40}  # other layers: every region
BOOTSTRAP = 2_000
LETTERS = "ABCDE"

SYSTEM = (
    "You are reading a map of a university's course catalog. Every region of the map has a short name that was "
    "written by a language model from the courses in it. You will be shown one region name and five candidate "
    "regions, each represented by five of its courses (title and the start of the description). Exactly one "
    "candidate is the region the name was written for. Score each candidate from 0 to 100 for how likely it is to "
    "be that region, judging only from the courses shown. Scores need not sum to 100."
)
SCHEMA = {
    "type": "object",
    "properties": {letter: {"type": "integer"} for letter in LETTERS},
    "required": list(LETTERS),
    "additionalProperties": False,
}


# ── item construction ───────────────────────────────────────────────────────


def nearest_regions(centroids: np.ndarray, cluster: int, k: int) -> list[int]:
    """The k regions whose centroids are nearest to `cluster`'s by cosine (centroids unit-norm), self excluded."""
    sims = centroids @ centroids[cluster]
    sims[cluster] = -np.inf
    return [int(i) for i in np.argsort(-sims)[:k]]


def far_regions(centroids: np.ndarray, cluster: int, k: int, rng: np.random.Generator) -> list[int]:
    """k regions sampled from the farther half of the same layer, by centroid cosine."""
    sims = centroids @ centroids[cluster]
    others = np.array([i for i in range(len(sims)) if i != cluster])
    order = others[np.argsort(sims[others])]  # farthest first
    far_half = order[: max(k, len(order) // 2)]
    return [int(i) for i in rng.choice(far_half, k, replace=False)]


def namer_exemplars(pm: PublishedMap, layer: int) -> list[list[int]]:
    """Replay stage 04's exemplar selection: the courses the namer saw for each region of `layer`."""
    labels = pm.layers[layer].astype(np.int64)
    centroids = centroids_from_labels(labels, pm.X)  # unnormalised means, as Toponymy builds them
    _, indices = diverse_exemplars(
        cluster_label_vector=labels,
        objects=pm.texts,
        object_vectors=pm.X,
        centroid_vectors=centroids,
        n_exemplars=N_EXEMPLARS,
        diversify_alpha=1.0,
        verbose=False,
    )
    return [[int(i) for i in idx] for idx in indices]


def heldout_documents(
    members: np.ndarray, exemplars: list[int], n: int, rng: np.random.Generator
) -> tuple[list[int], int]:
    """n member indices, preferring courses the namer did not see; returns (indices, how many were exemplars)."""
    exemplar_set = set(exemplars)
    pool = [int(i) for i in members if int(i) not in exemplar_set]
    chosen = [int(i) for i in rng.choice(pool, min(n, len(pool)), replace=False)] if pool else []
    n_from_exemplars = 0
    if len(chosen) < n:  # small region: fill from the exemplars, least central first
        fill = [e for e in reversed(exemplars) if e in set(members.tolist())][: n - len(chosen)]
        chosen += fill
        n_from_exemplars = len(fill)
    return chosen, n_from_exemplars


def excerpt(text: str, n_chars: int = DOC_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n_chars else text[: n_chars - 1].rsplit(" ", 1)[0] + "…"


def build_items(pm: PublishedMap, sample_per_layer: dict[int, int], seed: int = SEED, quick: bool = False) -> dict:
    """Frozen lineups for a sample of regions per layer; documents and distractors fixed per region."""
    rng = np.random.default_rng(seed)
    docs: dict[str, dict] = {}  # "layer:cluster" -> shown courses, frozen
    items = []
    for layer in range(pm.n_layers):
        n = pm.n_regions(layer)
        if n <= K:
            continue
        exemplars = namer_exemplars(pm, layer)
        cents = pm.centroids(layer)
        cents_2d = pm.centroids(layer, "2d")
        cents_2d_norm = cents_2d  # euclidean neighbours in the layout, for the overlap statistic only
        chosen = sorted(int(c) for c in rng.choice(n, min(sample_per_layer.get(layer, n), n), replace=False))
        if quick:
            chosen = chosen[:1]
        for c in chosen:
            gold = nearest_regions(cents, c, K - 1)
            d2 = np.linalg.norm(cents_2d_norm - cents_2d_norm[c], axis=1)
            d2[c] = np.inf
            near_2d = set(int(i) for i in np.argsort(d2)[: K - 1])
            gimme = far_regions(cents, c, K - 1, rng)
            candidates = [c] + gold + gimme
            for cand in candidates:
                key = f"{layer}:{cand}"
                if key not in docs:
                    idx, n_ex = heldout_documents(pm.members(layer, cand), exemplars[cand], N_DOCS, rng)
                    docs[key] = {
                        "course_idx": idx,
                        "n_from_exemplars": n_ex,
                        "n_members": int(len(pm.members(layer, cand))),
                    }
            not_candidates = [i for i in range(n) if i not in set([c] + gold)]
            shuffled = int(rng.choice(not_candidates))
            items.append(
                {
                    "item": f"L{layer}C{c}",
                    "layer": layer,
                    "cluster": c,
                    "name": pm.name((layer, c)),
                    "gold_distractors": gold,
                    "gold_distractors_also_nearest_2d": sum(g in near_2d for g in gold),
                    "gimme_distractors": gimme,
                    "shuffled_name_cluster": shuffled,
                    "shuffled_name": pm.name((layer, shuffled)),
                }
            )
    return {"items": items, "documents": docs}


# ── the lineup itself ───────────────────────────────────────────────────────


def format_lineup(name: str, candidate_docs: list[list[str]]) -> str:
    lines = [f'Region name: "{name}"', ""]
    for letter, group in zip(LETTERS, candidate_docs):
        lines.append(f"Candidate {letter}:")
        lines += [f"  {j}. {d}" for j, d in enumerate(group, 1)]
        lines.append("")
    lines.append("Which candidate is the named region? Score every candidate from 0 to 100.")
    return "\n".join(lines)


def probability_mass(scores: list[float], true_pos: int) -> float:
    total = float(sum(scores))
    return scores[true_pos] / total if total > 0 else 1.0 / len(scores)


def top1_correct(scores: list[float], true_pos: int) -> bool:
    best = max(scores)
    return scores[true_pos] == best and scores.count(best) == 1


def plan_calls(frozen: dict, pm: PublishedMap, seed: int = SEED) -> list[dict]:
    """Every listener call for the run: condition, sample, shown order (reshuffled per call), prompt."""
    rng = np.random.default_rng(seed + 1)
    docs = frozen["documents"]

    def shown(layer: int, cluster: int) -> list[str]:
        return [excerpt(pm.texts[i]) for i in docs[f"{layer}:{cluster}"]["course_idx"]]

    calls = []
    for it in frozen["items"]:
        layer, c = it["layer"], it["cluster"]
        conditions = [("gold", it["name"], it["gold_distractors"], s) for s in range(GOLD_SAMPLES)]
        conditions.append(("shuffled", it["shuffled_name"], it["gold_distractors"], 0))
        conditions.append(("gimme", it["name"], it["gimme_distractors"], 0))
        for condition, name, distractors, sample in conditions:
            candidates = [c] + list(distractors)
            order = [int(i) for i in rng.permutation(K)]  # position -> index into candidates
            shown_clusters = [candidates[i] for i in order]
            calls.append(
                {
                    "id": f"{it['item']}:{condition}:{sample}",
                    "item": it["item"],
                    "layer": layer,
                    "cluster": c,
                    "condition": condition,
                    "sample": sample,
                    "name_shown": name,
                    "shown_clusters": shown_clusters,
                    "true_pos": shown_clusters.index(c),
                    "user": format_lineup(name, [shown(layer, k) for k in shown_clusters]),
                }
            )
    return calls


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n: int = BOOTSTRAP) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    means = rng.choice(values, (n, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n": int(len(values)),
    }


def summarise(rows: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Per layer and overall: gold mass and top-1 (item means over samples), floor, ceiling, repeat band."""
    out = {}
    gold = rows[rows["condition"] == "gold"]
    per_item = gold.groupby(["item", "layer"], as_index=False).agg(pm=("pm", "mean"), top1=("top1", "mean"))
    # repeat band: p90 of |pm difference| between two samples of the same item
    wide = gold.pivot(index="item", columns="sample", values="pm")
    diffs = np.abs(wide.iloc[:, 0] - wide.iloc[:, 1]).to_numpy() if wide.shape[1] >= 2 else np.array([])
    for scope, sel in [
        ("overall", per_item),
        *[(f"layer_{L}", per_item[per_item["layer"] == L]) for L in sorted(per_item["layer"].unique())],
    ]:
        items = set(sel["item"])
        out[scope] = {
            "n_items": int(len(sel)),
            "gold_pm": bootstrap_mean(sel["pm"].to_numpy(), rng),
            "gold_top1": float(sel["top1"].mean()),
            "shuffled_pm": bootstrap_mean(
                rows[(rows["condition"] == "shuffled") & rows["item"].isin(items)]["pm"].to_numpy(), rng
            ),
            "gimme_pm": bootstrap_mean(
                rows[(rows["condition"] == "gimme") & rows["item"].isin(items)]["pm"].to_numpy(), rng
            ),
            "share_items_below_chance": float((sel["pm"] < 1 / K).mean()),
        }
    out["repeat_band_p90"] = float(np.percentile(diffs, 90)) if len(diffs) else None
    out["chance"] = 1 / K
    pos = rows[rows["condition"] == "gold"]
    mean_by_position = [float(np.mean([s[i] for s in pos["scores"]])) for i in range(K)]
    out["gold_mean_score_by_position"] = mean_by_position
    out["all_zero_rows"] = int((rows["scores"].map(sum) == 0).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--dry-run", action="store_true", help="build and write the items, estimate the calls, no listener")
    ap.add_argument("--quick", action="store_true", help="one region per layer, for a smoke test of the listener")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = config.source_paths(args.source)
    base = paths["base"]
    t_start = time.time()

    pm = load_published(args.source)
    print(f"{len(pm.course_id)} courses, layers {[pm.n_regions(i) for i in range(pm.n_layers)]}")
    frozen = build_items(pm, SAMPLE_PER_LAYER, quick=args.quick)
    calls = plan_calls(frozen, pm)
    n_leaky = sum(d["n_from_exemplars"] > 0 for d in frozen["documents"].values())
    approx_tokens = sum(len(SYSTEM) + len(c["user"]) for c in calls) / 3.6
    print(
        f"{len(frozen['items'])} items, {len(frozen['documents'])} regions shown as candidates "
        f"({n_leaky} too small to fill five documents without exemplars), {len(calls)} listener calls, "
        f"~{approx_tokens / 1e6:.2f}M input tokens"
    )
    if not args.quick:
        atomic_write_text(base / "wayfinding_items.json", json.dumps(frozen, indent=1, ensure_ascii=False))
    if args.dry_run:
        print("dry run: wrote wayfinding_items.json, no listener calls")
        return
    if not have_api_key():
        sys.exit("ANTHROPIC_API_KEY is not set; run with --dry-run or set the key")

    listener = Listener(cache_path=None if args.quick else base / "wayfinding_calls.jsonl")
    answers = listener.ask_many(
        [(c["id"], SYSTEM, c["user"], SCHEMA) for c in calls], workers=args.workers, label="lineup "
    )
    listener.close()
    rows, refusals = [], []
    for call, ans in zip(calls, answers):
        if refused(ans):
            refusals.append(
                {"id": call["id"], "item": call["item"], "name": call["name_shown"], "category": ans["refused"]}
            )
            continue
        scores = [max(0, min(100, int(ans[letter]))) for letter in LETTERS]
        rows.append(
            {
                "item": call["item"],
                "layer": call["layer"],
                "cluster": call["cluster"],
                "condition": call["condition"],
                "sample": call["sample"],
                "name_shown": call["name_shown"],
                "shown_clusters": json.dumps(call["shown_clusters"]),
                "true_pos": call["true_pos"],
                "scores": scores,
                "pm": probability_mass(scores, call["true_pos"]),
                "top1": top1_correct(scores, call["true_pos"]),
            }
        )
    rows = pd.DataFrame(rows)
    if refusals:
        refused_items = sorted({r["item"] for r in refusals})
        print(
            f"{len(refusals)} calls refused by the listener on {len(refused_items)} items: {refused_items}; those items are dropped"
        )
        rows = rows[~rows["item"].isin(refused_items)]
    summary = summarise(rows, np.random.default_rng(SEED))
    print(json.dumps({k: v for k, v in summary.items() if k.startswith(("overall", "layer"))}, indent=1))
    print(
        f"repeat band p90 {summary['repeat_band_p90']}, mean score by position {summary['gold_mean_score_by_position']}"
    )
    print(f"listener: {listener.describe()}")
    if listener.usage["calls"] < len(calls):
        print(f"{len(calls) - listener.usage['calls']} answers came from the cache of an earlier run")
    if args.quick:
        return

    out_rows = rows.copy()
    out_rows["scores"] = out_rows["scores"].map(json.dumps)
    out_rows.to_csv(base / "wayfinding_lineup.csv", index=False)
    items_df = pd.DataFrame(frozen["items"])
    docs = frozen["documents"]
    leak_items = [
        it["item"] for it in frozen["items"] if docs[f"{it['layer']}:{it['cluster']}"]["n_from_exemplars"] > 0
    ]
    record = {
        "what": "wayfinding lineup: probability mass a listener puts on the true region given its name, among the "
        "region and its four nearest same-layer neighbours, each shown as five held-out member courses",
        "protocol": "Steven's Toponymy label-quality instrument (TutteInstitute/toponymy discussion 177), applied to "
        "the published map's names as-is; the seed layouts from step 1 play no role because the task shows course "
        "text, not positions",
        "sample": {
            "per_layer": {
                f"layer_{L}": int((items_df["layer"] == L).sum()) for L in sorted(items_df["layer"].unique())
            },
            "rule": "forty regions at layers 0 and 1, every region at layers 2 and 3; seed " + str(SEED),
        },
        "candidates": {
            "k": K,
            "gold_distractors": "four nearest same-layer regions by centroid cosine in the embedding space",
            "gimme_distractors": "four regions sampled from the farther half of the layer by centroid cosine",
            "shuffled_name": "the name of a random same-layer region that is not among the five candidates",
            "share_gold_distractors_also_among_4_nearest_in_layout": float(
                items_df["gold_distractors_also_nearest_2d"].sum() / (4 * len(items_df))
            ),
        },
        "documents": {
            "per_candidate": N_DOCS,
            "text": f"title and description as embedded (embed_text), truncated to {DOC_CHARS} characters; no course codes",
            "held_out": f"Toponymy's central exemplar selection (n_exemplars={N_EXEMPLARS}, diversify_alpha=1.0) replayed "
            "on the published labels and embeddings; those courses are excluded from the shown sample",
            "regions_filled_from_exemplars": n_leaky,
            "items_whose_true_region_shows_exemplars": leak_items,
            "frozen": "one document sample per region, reused wherever the region appears; candidate order reshuffled per call",
        },
        "conditions": {"gold_samples": GOLD_SAMPLES, "shuffled_samples": 1, "gimme_samples": 1},
        "refusals": {
            "what": "calls the listener's safety classifier declined; every condition of an affected item is dropped from "
            "the results so that no item is scored on a subset of its conditions",
            "calls": refusals,
            "items_dropped": sorted({r["item"] for r in refusals}),
        },
        "metric": "probability mass s_true / sum(s), uniform 1/k when every score is zero; top-1 is descriptive "
        "(true region has the unique highest score)",
        "results": summary,
        "listener": listener.describe(),
        "prompt": {"system": SYSTEM, "user_example": calls[0]["user"][:1200] + " …", "schema": SCHEMA},
        "disclosures": [
            "the names were written by Claude Opus 5 (stage 04) and are scored here by Claude Sonnet 5, a different "
            "model of the same family; a same-model listener was avoided, a same-family one was not",
            "the listener is an LLM; its agreement with a human on the name-intrusion items (name_intrusion.json) is "
            "the grounding for treating it as a judge",
            "held-out documents are a replay of the exemplar selection, not a log of the namer's actual prompts",
        ],
        "minutes": round((time.time() - t_start) / 60, 1),
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_text(base / "wayfinding_lineup.json", json.dumps(record, indent=1, ensure_ascii=False))
    print(f"wrote wayfinding_items.json, wayfinding_lineup.csv, wayfinding_lineup.json; {record['minutes']} min")


if __name__ == "__main__":
    main()
