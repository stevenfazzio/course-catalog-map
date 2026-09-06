import numpy as np
import pytest

import name_intrusion as ni
from published_map import PublishedMap


def _tree_map() -> PublishedMap:
    """Layer 1 parents 0, 1, 2 with 5, 4, 6 finest children; parents 0 and 1 share a coarsest ancestor, 2 does not."""
    kids = {0: 5, 1: 4, 2: 6}
    layer0 = np.concatenate([np.full(3, c) for c in range(sum(kids.values()))]).astype(np.int32)
    parent_of_child = {}
    c = 0
    for p, k in kids.items():
        for _ in range(k):
            parent_of_child[(0, c)] = (1, p)
            c += 1
    layer1 = np.array([parent_of_child[(0, int(x))][1] for x in layer0], dtype=np.int32)
    parent = parent_of_child | {(1, 0): (2, 0), (1, 1): (2, 0), (1, 2): (2, 1), (2, 0): (3, 0), (2, 1): (3, 0)}
    n = len(layer0)
    X = np.random.default_rng(0).normal(size=(n, 4))
    return PublishedMap(
        course_id=np.arange(n),
        texts=[f"t{i}" for i in range(n)],
        X=X / np.linalg.norm(X, axis=1, keepdims=True),
        Y=X[:, :2],
        layers=[layer0, layer1, np.array([0 if p < 2 else 1 for p in layer1], dtype=np.int32)],
        names=[[f"fine {c}" for c in range(15)], ["P0", "P1", "P2"], ["A", "B"]],
        parent=parent,
    )


def test_sibling_groups_and_eligible_intruders():
    pm = _tree_map()
    groups = ni.sibling_groups(pm)
    assert [(g[0], g[1], len(g[2])) for g in groups] == [((1, 0), 0, 5), ((1, 1), 0, 4), ((1, 2), 0, 6)]
    # P0's children are excluded, and so are P1's (same coarsest ancestor A); only P2's remain
    assert ni.eligible_intruders(pm, (1, 0), 0) == list(range(9, 15))
    assert ni.eligible_intruders(pm, (1, 2), 0) == list(range(0, 9))


def test_build_items_structure_and_limits():
    pm = _tree_map()
    items = ni.build_items(pm, n_items=3)
    assert [it["item"] for it in items] == ["I01", "I02", "I03"]
    assert {tuple(it["ancestor"]) for it in items} == {(1, 0), (1, 1), (1, 2)}
    for it in items:
        parent = tuple(it["ancestor"])
        assert it["unit"] == "parent"
        assert len(it["siblings"]) == 4 and all(pm.parent[(0, c)] == parent for c in it["siblings"])
        assert pm.parent[(0, it["intruder"])] != parent
        assert pm.coarsest_ancestor((0, it["intruder"])) != pm.coarsest_ancestor(parent)
        assert sorted(it["clusters_in_order"]) == sorted(it["siblings"] + [it["intruder"]])
        assert it["names_in_order"][ni.LETTERS.index(it["intruder_pos"])] == pm.name((0, it["intruder"]))
    # the fourth item is the one grandparent group: A's nine finest descendants through P0 and P1; B's six come
    # through P2 alone and would duplicate a parent group, so B is not a grandparent group
    assert [(g[0], g[1], len(g[2])) for g in ni.grandparent_groups(pm)] == [((2, 0), 0, 9)]
    four = ni.build_items(pm, n_items=4)
    assert four[3]["unit"] == "grandparent" and four[3]["ancestor"] == [2, 0]
    assert all(c < 9 for c in four[3]["siblings"]) and four[3]["intruder"] >= 9
    with pytest.raises(ValueError):  # no parent has eight children, so no disjoint second draw exists either
        ni.build_items(pm, n_items=5)
    table = ni.items_table(items)
    assert len(table) == 15 and set(table["position"]) == set(ni.LETTERS)


def test_score_and_wilson():
    pm = _tree_map()
    items = ni.build_items(pm, n_items=3)
    picks = {it["item"]: it["intruder_pos"] for it in items}
    perfect = ni.score(items, picks)
    assert perfect["n_correct"] == 3 and perfect["accuracy"] == 1.0 and perfect["misses"] == []
    assert perfect["p_value_vs_chance"] == pytest.approx(0.2**3)
    wrong = dict(picks)
    wrong[items[0]["item"]] = next(x for x in ni.LETTERS if x != items[0]["intruder_pos"])
    res = ni.score(items, wrong)
    assert res["n_correct"] == 2 and len(res["misses"]) == 1 and res["misses"][0]["item"] == items[0]["item"]
    assert res["by_unit"]["parent"]["n"] == 3
    lo, hi = ni.wilson_interval(40, 50)
    assert 0.66 < lo < 0.8 < hi < 0.9
    assert ni.wilson_interval(0, 0) == pytest.approx([float("nan"), float("nan")], nan_ok=True)


def test_read_answers_accepts_letters_and_digits(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("item,choice\nI01,B\nI02,3\n")
    assert ni.read_answers(p) == {"I01": "B", "I02": "C"}
