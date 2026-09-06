import numpy as np

import wayfinding_lineup as wl
from published_map import PublishedMap


def _synthetic_map(n_regions=8, per=12, dim=8, seed=0) -> PublishedMap:
    """Eight tight finest regions in 8-d, grouped four and four at the coarser layer, under one root."""
    rng = np.random.default_rng(seed)
    centres = np.eye(dim)[:n_regions]
    X = np.concatenate([c + 0.05 * rng.normal(size=(per, dim)) for c in centres])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    layer0 = np.repeat(np.arange(n_regions), per).astype(np.int32)
    layer1 = (layer0 >= n_regions // 2).astype(np.int32)
    n = len(X)
    parent = {(0, c): (1, int(c >= n_regions // 2)) for c in range(n_regions)} | {(1, 0): (2, 0), (1, 1): (2, 0)}
    return PublishedMap(
        course_id=np.arange(n),
        texts=[f"Course {i}\nAbout region {layer0[i]} " + "word " * 200 for i in range(n)],
        X=X,
        Y=X[:, :2].copy(),
        layers=[layer0, layer1],
        names=[[f"Region {c}" for c in range(n_regions)], ["Left half", "Right half"]],
        parent=parent,
    )


def test_nearest_and_far_regions_use_centroid_cosine():
    cents = np.eye(6)
    cents[1] = [0.9, 0.436, 0, 0, 0, 0]  # region 1 leans towards region 0
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    near = wl.nearest_regions(cents, 0, 2)
    assert near[0] == 1 and 0 not in near and len(near) == 2
    rng = np.random.default_rng(0)
    far = wl.far_regions(cents, 0, 2, rng)
    assert 0 not in far and 1 not in far and len(set(far)) == 2  # region 1 is in the near half


def test_heldout_documents_exclude_exemplars_and_fill_only_when_short():
    rng = np.random.default_rng(0)
    members = np.arange(20)
    exemplars = [0, 1, 2, 3, 4, 5, 6, 7]
    docs, n_ex = wl.heldout_documents(members, exemplars, 5, rng)
    assert n_ex == 0 and len(docs) == 5 and not set(docs) & set(exemplars)
    small = np.arange(10)  # ten members, eight exemplars: two held out, three filled from the least central
    docs, n_ex = wl.heldout_documents(small, exemplars, 5, rng)
    assert len(docs) == 5 and n_ex == 3 and {8, 9} <= set(docs) and docs[-3:] == [7, 6, 5]


def test_probability_mass_and_top1():
    assert wl.probability_mass([10, 30, 0, 0, 0], 1) == 0.75
    assert wl.probability_mass([0, 0, 0, 0, 0], 2) == 0.2
    assert wl.top1_correct([10, 30, 0, 0, 0], 1) and not wl.top1_correct([30, 30, 0, 0, 0], 1)


def test_build_items_and_plan_calls_on_a_synthetic_map():
    pm = _synthetic_map()
    frozen = wl.build_items(pm, {0: 3})  # layer 1 has two regions, fewer than K, so it is skipped
    items, docs = frozen["items"], frozen["documents"]
    assert len(items) == 3 and all(it["layer"] == 0 for it in items)
    for it in items:
        cands = [it["cluster"]] + it["gold_distractors"]
        assert len(set(cands)) == wl.K and it["cluster"] not in it["gold_distractors"]
        assert it["shuffled_name_cluster"] not in cands
        assert len(set(it["gimme_distractors"])) == wl.K - 1 and it["cluster"] not in it["gimme_distractors"]
        for c in cands + it["gimme_distractors"]:
            d = docs[f"0:{c}"]
            assert len(d["course_idx"]) == wl.N_DOCS and set(d["course_idx"]) <= set(pm.members(0, c).tolist())
    # twelve members, eight exemplars: four held out, one filled from the exemplars
    assert all(d["n_from_exemplars"] == 1 for d in docs.values())
    calls = wl.plan_calls(frozen, pm)
    assert len(calls) == 3 * (wl.GOLD_SAMPLES + 2)
    for call in calls:
        assert call["shown_clusters"][call["true_pos"]] == call["cluster"]
        assert len(set(call["shown_clusters"])) == wl.K
        assert all(f"Candidate {letter}:" in call["user"] for letter in wl.LETTERS)
        assert f'Region name: "{call["name_shown"]}"' in call["user"]
    gold = [c for c in calls if c["item"] == items[0]["item"] and c["condition"] == "gold"]
    assert len({tuple(c["shown_clusters"]) for c in gold}) > 1  # candidate order is reshuffled per sample
    shuffled = next(c for c in calls if c["condition"] == "shuffled")
    assert shuffled["name_shown"] != next(i for i in items if i["item"] == shuffled["item"])["name"]


def test_namer_exemplars_are_central_members():
    pm = _synthetic_map()
    ex = wl.namer_exemplars(pm, 0)
    assert len(ex) == 8 and all(len(e) == wl.N_EXEMPLARS for e in ex)
    for c, idx in enumerate(ex):
        assert set(idx) <= set(pm.members(0, c).tolist())
