import numpy as np
import pandas as pd

import crosslisting_distance as cd
import department_centrality as dc
import linear_probe as lp


def test_shared_counts_and_pair_table():
    listed_in = [["A"], ["A", "B"], ["A", "B", "C"], ["B", "Z"]]  # Z has no centroid and is ignored
    shared = cd.shared_counts(listed_in, {"A", "B", "C"})
    assert shared == {("A", "B"): 2, ("A", "C"): 1, ("B", "C"): 1}
    C = np.eye(3)
    school = {"A": "s1", "B": "s1", "C": "s2"}
    df = cd.pair_table(["A", "B", "C"], C, C, school, {("A", "B"): 2})
    assert len(df) == 3 and df["linked"].tolist() == [True, False, False]
    assert df["same_school"].tolist() == [True, False, False]
    assert np.allclose(df["distance_heldout"], 1.0)


def test_prob_closer_and_permute_within():
    assert cd.prob_closer(np.array([0.1, 0.2]), np.array([0.3, 0.4])) == 1.0
    assert cd.prob_closer(np.array([0.3, 0.4]), np.array([0.1, 0.2])) == 0.0
    rng = np.random.default_rng(0)
    x = np.array([1, 0, 0, 1, 1, 1, 0, 0])
    strata = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    p = cd.permute_within(rng, x, strata)
    assert p[:4].sum() == 2 and p[4:].sum() == 2


def test_linked_effect_finds_a_planted_gap_and_none_for_random_links():
    rng = np.random.default_rng(3)
    n = 400
    strata = rng.choice(["p", "q", "r"], n)
    linked = rng.random(n) < 0.3
    base = np.where(strata == "p", 0.5, 0.3)
    y = base - 0.1 * linked + 0.02 * rng.normal(size=n)
    df = pd.DataFrame({"distance": y, "school_pair": strata, "linked": linked})
    eff = cd.linked_effect(df, "distance", n_perm=200)
    assert abs(eff["estimate"] + 0.1) < 0.01 and eff["permutation_p"] < 0.01
    df["distance"] = base + 0.02 * rng.normal(size=n)
    null = cd.linked_effect(df, "distance", n_perm=200)
    assert null["permutation_p"] > 0.05


def test_displacement_leave_one_out():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dept_index = np.array([0, 0, 0, 0])
    sums = X[:3].sum(axis=0, keepdims=True)  # the centroid sum over the three single-listed courses
    counts = np.array([3])
    d = cd.displacement_to(X, sums, counts, dept_index, leave_out=np.array([True, True, True, False]))
    assert np.allclose(d[:3], 0.0) and np.isclose(d[3], 1.0)
    assert np.isnan(cd.displacement_to(X, sums, counts, np.array([-1, 0, 0, 0]), np.zeros(4, bool))[0])


def test_probe_separates_synthetic_classes():
    rng = np.random.default_rng(0)
    centres = np.eye(3)
    y = np.repeat(["a", "b", "c"], 30)
    F = np.concatenate([c + 0.1 * rng.normal(size=(30, 3)) for c in centres])
    pred, folds = lp.probe("embedding", F, y, outer_folds=3, inner_folds=2, c_grid=[1.0], n_jobs=1)
    assert (pred == y).mean() > 0.95 and len(folds) == 3
    s = lp.scores(y, pred, folds)
    assert s["accuracy"] > 0.95 and s["macro_f1"] > 0.95 and s["chosen_C_by_fold"] == [1.0, 1.0, 1.0]
    texts = np.array([f"{c}word {c}word filler" for c in y], dtype=object)  # tokens need two characters
    pred_t, _ = lp.probe("tfidf", texts, y, outer_folds=3, inner_folds=2, c_grid=[0.1, 1.0], n_jobs=1)
    assert (pred_t == y).mean() == 1.0
    b = lp.baselines(y)
    assert b == {"n_classes": 3, "majority_class": 1 / 3, "frequency_matched_guess": 1 / 3}


def test_centralities_find_the_hub():
    # Node 0 is similar to everyone; 1-3 are similar only to node 0 and to their own neighbour in a chain.
    S = np.array([[1.0, 0.9, 0.9, 0.9], [0.9, 1.0, 0.3, 0.1], [0.9, 0.3, 1.0, 0.3], [0.9, 0.1, 0.3, 1.0]])
    c = dc.centralities(S, k=1)
    assert np.argmax(c["strength"]) == 0 and np.argmax(c["eigenvector"]) == 0
    assert c["indegree"][0] == 3.0 and c["indegree"].sum() == 4.0  # node 0 must point at one of its tied neighbours
    assert np.argmax(c["closeness"]) == 0 and c["closeness"][0] == 1.0
    assert dc.rank_desc(np.array([0.2, 0.9, 0.5])).tolist() == [3, 1, 2]
