"""Step 3 of docs/plan.md: the name-intrusion task, a human check on the LLM-written region names.

    uv run python experiments/name_intrusion.py            # build the fifty items, the rater page and the sealed key
    uv run python experiments/name_intrusion.py --llm      # the listener model answers the same items
    uv run python experiments/name_intrusion.py --score    # score the human answers (and the listener's), write the record

Chang et al.'s (2009) word-intrusion protocol adapted to region names. Each item shows five names from one layer of
the map: four share an ancestor region, one is the intruder, a same-layer region under a different coarsest
ancestor. The rater picks the one that does not belong. If the names carry the
content of their regions, siblings read as a group and the intruder stands out; a vague or wrong name makes the item
fail. Chance is 0.20. Items come from every parent with four or more children at one layer (40 on this map); the
rest are grandparent groups, four descendants at one layer of a region two or more layers up, drawn through at least
two of its children, chosen at random to reach fifty. Layers are mixed and both item order and the intruder's
position are shuffled.

The rater is the map's author, a single rater who had seen the map and every name before rating. Both facts go into
the record next to the result. The listener model (listener.py) answers the same fifty items so its agreement with
the human is on file: that is the grounding for using it as the judge in the wayfinding lineup.

Rating: serve data/<source>/ with `python3 -m http.server 8765 --bind 127.0.0.1`, open name_intrusion_rater.html,
answer every item, copy the answers and save them as data/<source>/name_intrusion_answers.csv, then run --score.
Do not open name_intrusion_key.json before rating.

Writes to data/<source>/ (stage 07 copies them into the record):
    name_intrusion_items.csv    item, position (A-E), name: what the rater sees
    name_intrusion_key.json     the intruder position and the regions behind every item (the sealed key)
    name_intrusion_rater.html   the rating page
    name_intrusion_llm.csv      the listener's picks (--llm)
    name_intrusion.json         the scored result with disclosures (--score)
"""

import argparse
import datetime as dt
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "experiments"))
import config  # noqa: E402
from io_utils import atomic_write_text  # noqa: E402
from listener import Listener, have_api_key, refused  # noqa: E402
from published_map import PublishedMap, Region, load_published  # noqa: E402

N_ITEMS = 50
N_SIBLINGS = 4
SEED = 0
LETTERS = "ABCDE"
CHANCE = 1 / (N_SIBLINGS + 1)

INSTRUCTIONS = (
    "Each item shows five names of regions on a map of Stanford's course catalog. Four of them name regions that "
    "sit together inside one larger region of the map; one comes from elsewhere on the map. Pick the one that does "
    "not belong."
)
LLM_SYSTEM = (
    "You are shown five names of regions on a map of a university's course catalog. The names were written by a "
    "language model from the courses in each region. Four of the five name regions that sit together inside one "
    "larger region of the map; one comes from elsewhere on the map. Pick the one that does not belong."
)
LLM_SCHEMA = {
    "type": "object",
    "properties": {"intruder": {"type": "string", "enum": list(LETTERS)}},
    "required": ["intruder"],
    "additionalProperties": False,
}


# ── items ────────────────────────────────────────────────────────────────────


def sibling_groups(pm: PublishedMap, min_children: int = N_SIBLINGS) -> list[tuple[Region, int, list[int]]]:
    """(parent, child layer, child cluster ids) for every named parent with `min_children`+ children at one layer."""
    groups = []
    for parent in sorted(set(pm.parent.values()) - {pm.root}):
        kids = pm.children(parent)
        for layer in sorted({k[0] for k in kids}):
            same = sorted(c for (lyr, c) in kids if lyr == layer)
            if len(same) >= min_children:
                groups.append((parent, layer, same))
    return groups


def grandparent_groups(pm: PublishedMap, min_members: int = N_SIBLINGS) -> list[tuple[Region, int, list[int]]]:
    """(ancestor, descendant layer, cluster ids) for named regions two or more layers up with `min_members`+
    descendants at one layer drawn from at least two of the ancestor's children (one child would be a parent group)."""
    groups = []
    for ancestor in sorted(set(pm.parent.values()) - {pm.root}):
        for layer in range(0, ancestor[0] - 1):
            desc = pm.descendants(ancestor, layer)
            via = [
                kid
                for kid in pm.children(ancestor)
                if kid[0] == layer or (kid[0] > layer and pm.descendants(kid, layer))
            ]
            if len(desc) >= min_members and len(via) >= 2:
                groups.append((ancestor, layer, desc))
    return groups


def eligible_intruders(pm: PublishedMap, ancestor: Region, layer: int) -> list[int]:
    """Same-layer regions under a different coarsest ancestor than `ancestor`'s (so never its descendants or cousins)."""
    top = pm.coarsest_ancestor(ancestor)
    return [c for c in range(pm.n_regions(layer)) if pm.coarsest_ancestor((layer, c)) != top]


def build_items(pm: PublishedMap, n_items: int = N_ITEMS, seed: int = SEED) -> list[dict]:
    """Every parent group once; then grandparent groups at random until `n_items`; then disjoint second draws."""
    rng = np.random.default_rng(seed)
    parents = sibling_groups(pm)
    grand = grandparent_groups(pm)
    draws: list[tuple[str, Region, int, list[int]]] = []
    used: dict[tuple[Region, int], set[int]] = {}
    for i in rng.permutation(len(parents)):
        ancestor, layer, kids = parents[i]
        pick = sorted(int(c) for c in rng.choice(kids, N_SIBLINGS, replace=False))
        draws.append(("parent", ancestor, layer, pick))
        used[(ancestor, layer)] = set(pick)
    for i in rng.permutation(len(grand)):
        if len(draws) >= n_items:
            break
        ancestor, layer, desc = grand[i]
        pick = sorted(int(c) for c in rng.choice(desc, N_SIBLINGS, replace=False))
        draws.append(("grandparent", ancestor, layer, pick))
    for ancestor, layer, kids in sorted(parents, key=lambda g: -len(g[2])):
        if len(draws) >= n_items:
            break
        rest = [c for c in kids if c not in used[(ancestor, layer)]]
        if len(rest) >= N_SIBLINGS:
            pick = sorted(int(c) for c in rng.choice(rest, N_SIBLINGS, replace=False))
            draws.append(("parent", ancestor, layer, pick))
            used[(ancestor, layer)] |= set(pick)
    if len(draws) < n_items:
        raise ValueError(f"only {len(draws)} items possible with {N_SIBLINGS} siblings; {n_items} asked")
    draws = draws[:n_items]
    items = []
    for k, (unit, ancestor, layer, pick) in enumerate(draws, 1):
        intruder = int(rng.choice(eligible_intruders(pm, ancestor, layer)))
        shown = pick + [intruder]
        pos = [int(i) for i in rng.permutation(len(shown))]  # position -> index into shown
        clusters_in_order = [shown[i] for i in pos]
        intruder_parent = pm.parent[(layer, intruder)]
        items.append(
            {
                "item": f"I{k:02d}",
                "child_layer": layer,
                "unit": unit,
                "ancestor": list(ancestor),
                "ancestor_name": pm.name(ancestor),
                "siblings": pick,
                "intruder": intruder,
                "intruder_parent_name": pm.name(intruder_parent) if intruder_parent != pm.root else "(root)",
                "clusters_in_order": clusters_in_order,
                "names_in_order": [pm.name((layer, c)) for c in clusters_in_order],
                "intruder_pos": LETTERS[clusters_in_order.index(intruder)],
            }
        )
    return items


def items_table(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": it["item"], "position": letter, "name": name}
            for it in items
            for letter, name in zip(LETTERS, it["names_in_order"])
        ]
    )


def rater_page(items: list[dict], answers_file: str) -> str:
    blocks = []
    for it in items:
        options = "".join(
            f'<label><input type="radio" name="{it["item"]}" value="{letter}"> <b>{letter}</b> {html.escape(name)}</label>'
            for letter, name in zip(LETTERS, it["names_in_order"])
        )
        blocks.append(f'<fieldset id="{it["item"]}"><legend>{it["item"]}</legend>{options}</fieldset>')
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Name intrusion task</title>
<style>
html{{color-scheme:light;background:#fff}}body{{font:15px/1.45 -apple-system,system-ui,sans-serif;max-width:720px;margin:2em auto;padding:0 1em;color:#1f2328;background:#fff}}
fieldset{{border:1px solid #d0d7de;border-radius:6px;margin:0 0 1em;padding:.6em 1em}}legend{{color:#57606a;font-size:12px}}
label{{display:block;padding:.15em 0;cursor:pointer}}fieldset.done{{border-color:#2da44e}}
textarea{{width:100%;height:9em;font:12px monospace}}button{{font-size:15px;padding:.4em .9em}}
.bar{{position:sticky;top:0;background:#fff;padding:.5em 0;border-bottom:1px solid #d0d7de}}
</style></head><body>
<h1>Name intrusion task</h1>
<p>{html.escape(INSTRUCTIONS)} There are {len(items)} items; answers are kept in this browser until you copy them.</p>
<div class="bar"><span id="progress"></span> &nbsp; <button id="copy">Copy answers</button>
<span id="note"></span></div>
{"".join(blocks)}
<p>Answers as CSV (save as <code>{html.escape(answers_file)}</code>, then run <code>name_intrusion.py --score</code>):</p>
<textarea id="out" readonly></textarea>
<script>
const items = {json.dumps([it["item"] for it in items])};
const key = "name_intrusion_answers";
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(key) || "{{}}"); }} catch (e) {{ state = {{}}; }}
function render() {{
  let n = 0;
  for (const it of items) {{
    const fs = document.getElementById(it);
    if (state[it]) {{ n++; fs.classList.add("done"); const r = fs.querySelector(`input[value="${{state[it]}}"]`); if (r) r.checked = true; }}
  }}
  document.getElementById("progress").textContent = `${{n}} of ${{items.length}} answered`;
  document.getElementById("out").value = "item,choice\\n" + items.filter(it => state[it]).map(it => `${{it}},${{state[it]}}`).join("\\n") + "\\n";
}}
document.querySelectorAll("input[type=radio]").forEach(r => r.addEventListener("change", e => {{
  state[e.target.name] = e.target.value;
  try {{ localStorage.setItem(key, JSON.stringify(state)); }} catch (err) {{}}
  render();
}}));
document.getElementById("copy").addEventListener("click", async () => {{
  const text = document.getElementById("out").value;
  try {{ await navigator.clipboard.writeText(text); document.getElementById("note").textContent = "copied"; }}
  catch (e) {{ document.getElementById("note").textContent = "select the text box below and copy"; }}
}});
render();
</script></body></html>
"""


# ── scoring ──────────────────────────────────────────────────────────────────


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def read_answers(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"item", "choice"} <= set(df.columns), f"{path} needs item,choice columns"
    out = {}
    for item, choice in zip(df["item"].str.strip(), df["choice"].str.strip().str.upper()):
        if choice.isdigit():
            choice = LETTERS[int(choice) - 1]
        assert choice in LETTERS, f"{path}: bad choice {choice!r} for {item}"
        out[item] = choice
    return out


def score(items: list[dict], picks: dict[str, str]) -> dict:
    """Accuracy overall and per child layer, Wilson 95% intervals, exact binomial test against chance."""
    rows = [(it, picks.get(it["item"])) for it in items]
    answered = [(it, p) for it, p in rows if p is not None]
    correct = [p == it["intruder_pos"] for it, p in answered]
    out = {
        "n_items": len(items),
        "n_answered": len(answered),
        "n_correct": int(sum(correct)),
        "accuracy": float(np.mean(correct)) if answered else float("nan"),
        "wilson95": wilson_interval(int(sum(correct)), len(answered)),
        "p_value_vs_chance": float(binomtest(int(sum(correct)), len(answered), CHANCE, alternative="greater").pvalue)
        if answered
        else float("nan"),
        "chance": CHANCE,
        "by_child_layer": {},
        "misses": [
            {
                "item": it["item"],
                "child_layer": it["child_layer"],
                "unit": it["unit"],
                "ancestor_name": it["ancestor_name"],
                "picked": it["names_in_order"][LETTERS.index(p)],
                "intruder": it["names_in_order"][LETTERS.index(it["intruder_pos"])],
                "intruder_parent_name": it["intruder_parent_name"],
            }
            for it, p in answered
            if p != it["intruder_pos"]
        ],
    }

    def subset(pred) -> dict:
        sel = [(it, p) for it, p in answered if pred(it)]
        c = [p == it["intruder_pos"] for it, p in sel]
        return {
            "n": len(sel),
            "n_correct": int(sum(c)),
            "accuracy": float(np.mean(c)) if sel else float("nan"),
            "wilson95": wilson_interval(int(sum(c)), len(sel)),
        }

    for layer in sorted({it["child_layer"] for it in items}):
        out["by_child_layer"][f"layer_{layer}"] = subset(lambda it, L=layer: it["child_layer"] == L)
    out["by_unit"] = {u: subset(lambda it, u=u: it["unit"] == u) for u in sorted({it["unit"] for it in items})}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--llm", action="store_true", help="the listener answers the items")
    ap.add_argument("--score", action="store_true", help="score the answers files and write name_intrusion.json")
    ap.add_argument(
        "--answers", default=None, help="human answers CSV (default data/<source>/name_intrusion_answers.csv)"
    )
    args = ap.parse_args()
    base = config.source_paths(args.source)["base"]
    items_path, key_path = base / "name_intrusion_items.csv", base / "name_intrusion_key.json"
    answers_path = Path(args.answers) if args.answers else base / "name_intrusion_answers.csv"
    llm_path = base / "name_intrusion_llm.csv"

    if not (args.llm or args.score):
        pm = load_published(args.source)
        groups = sibling_groups(pm)
        print(
            f"{len(groups)} parents with {N_SIBLINGS}+ children at one layer, {len(grandparent_groups(pm))} grandparent groups"
        )
        items = build_items(pm)
        table = items_table(items)
        table.to_csv(items_path, index=False)
        atomic_write_text(
            key_path,
            json.dumps(
                {
                    "instructions": INSTRUCTIONS,
                    "construction": f"four regions at one layer that share an ancestor (children of one parent where the map "
                    f"has enough of them, otherwise descendants of one grandparent through at least two of its children) plus "
                    f"one same-layer region under a different coarsest ancestor; every parent with {N_SIBLINGS}+ children once, "
                    f"then grandparent groups at random to fill; item order and intruder position shuffled; seed {SEED}",
                    "n_items": len(items),
                    "by_child_layer": {
                        f"layer_{L}": sum(it["child_layer"] == L for it in items)
                        for L in sorted({it["child_layer"] for it in items})
                    },
                    "by_unit": {u: sum(it["unit"] == u for it in items) for u in sorted({it["unit"] for it in items})},
                    "items": items,
                },
                indent=1,
                ensure_ascii=False,
            ),
        )
        atomic_write_text(base / "name_intrusion_rater.html", rater_page(items, str(answers_path)))
        by_layer = pd.Series([it["child_layer"] for it in items]).value_counts().sort_index().to_dict()
        print(
            f"{len(items)} items, by child layer {by_layer}; intruder positions {pd.Series([it['intruder_pos'] for it in items]).value_counts().sort_index().to_dict()}"
        )
        print(f"wrote {items_path.name}, {key_path.name}, name_intrusion_rater.html")
        return

    key = json.loads(key_path.read_text())
    items = key["items"]

    if args.llm:
        if not have_api_key():
            sys.exit("ANTHROPIC_API_KEY is not set")
        listener = Listener(cache_path=base / "name_intrusion_calls.jsonl")
        questions = [
            (
                it["item"],
                LLM_SYSTEM,
                "\n".join(f"{letter}. {name}" for letter, name in zip(LETTERS, it["names_in_order"]))
                + "\n\nWhich one does not belong?",
                LLM_SCHEMA,
            )
            for it in items
        ]
        answers = listener.ask_many(questions, workers=8, label="intrusion ")
        listener.close()
        picks = {it["item"]: a["intruder"] for it, a in zip(items, answers) if not refused(a)}
        if len(picks) < len(items):
            print(f"{len(items) - len(picks)} items refused by the listener; left unanswered")
        pd.DataFrame({"item": list(picks), "choice": list(picks.values())}).to_csv(llm_path, index=False)
        res = score(items, picks)
        print(
            f"listener: {res['n_correct']}/{res['n_answered']} correct ({res['accuracy']:.2f}); {listener.describe()}"
        )
        print(f"wrote {llm_path.name}")
        return

    if args.score:
        record = {
            "what": "name-intrusion task on the published region names: four children of one region plus one intruder "
            "from elsewhere, pick the intruder; Chang et al. 2009 adapted to region names",
            "construction": key["construction"],
            "instructions_shown": INSTRUCTIONS,
            "n_items": key["n_items"],
            "by_child_layer": key["by_child_layer"],
            "by_unit": key.get("by_unit"),
        }
        human = read_answers(answers_path) if answers_path.exists() else None
        llm = read_answers(llm_path) if llm_path.exists() else None
        if human is None and llm is None:
            sys.exit(f"nothing to score: neither {answers_path} nor {llm_path} exists")
        record["status"] = "complete" if human is not None else "awaiting the human rating (name_intrusion_answers.csv)"
        if human is not None:
            record["human"] = score(items, human) | {
                "rater": "the map's author, a single rater, who had seen the map and every region name before rating",
            }
            print(
                f"human: {record['human']['n_correct']}/{record['human']['n_answered']} correct, "
                f"Wilson 95% {[round(x, 2) for x in record['human']['wilson95']]}, p vs chance {record['human']['p_value_vs_chance']:.2g}"
            )
        if llm is not None:
            record["listener"] = score(items, llm) | {
                "model": json.loads((base / "name_intrusion_calls.jsonl").read_text().splitlines()[0])["model"]
            }
            print(f"listener: {record['listener']['n_correct']}/{record['listener']['n_answered']} correct")
        if human is not None and llm is not None:
            both = [it for it in items if it["item"] in human and it["item"] in llm]
            agree = [human[it["item"]] == llm[it["item"]] for it in both]
            hc = [human[it["item"]] == it["intruder_pos"] for it in both]
            lc = [llm[it["item"]] == it["intruder_pos"] for it in both]
            record["human_listener_agreement"] = {
                "n": len(both),
                "same_pick": int(sum(agree)),
                "share_same_pick": float(np.mean(agree)) if both else float("nan"),
                "both_correct": int(sum(h and m for h, m in zip(hc, lc))),
                "human_only_correct": int(sum(h and not m for h, m in zip(hc, lc))),
                "listener_only_correct": int(sum(m and not h for h, m in zip(hc, lc))),
                "both_wrong": int(sum(not h and not m for h, m in zip(hc, lc))),
            }
            print(f"agreement: same pick on {sum(agree)}/{len(both)}")
        record["disclosures"] = [
            "single human rater, the map's author, who had seen the map and all names before rating",
            "the region names were written by Claude Opus 5 (stage 04); the listener is Claude Sonnet 5",
            "items test whether names of regions that share an ancestor read as one group; a miss can come from a vague "
            "or wrong name or from an ancestor whose descendants do not read as one group",
        ]
        record["computed_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        atomic_write_text(base / "name_intrusion.json", json.dumps(record, indent=1, ensure_ascii=False))
        print("wrote name_intrusion.json")


if __name__ == "__main__":
    main()
