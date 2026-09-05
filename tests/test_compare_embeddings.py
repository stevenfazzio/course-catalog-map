import numpy as np
import pandas as pd

import compare_embeddings as ce


def _clustered(n_per=150, n_clusters=3, dim=16, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    X = np.concatenate([c + noise * rng.normal(size=(n_per, dim)) for c in centres])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = [frozenset([f"c{i}"]) for i in range(n_clusters) for _ in range(n_per)]
    return X, labels


def test_knn_excludes_self_even_with_duplicates():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    neigh = ce.knn_indices(X, 1)
    assert neigh.shape == (4, 1)
    assert all(neigh[i, 0] != i for i in range(4))


def test_agreement_separates_structure_from_noise():
    X, labels = _clustered()
    good = ce.agreement_for_label(X, labels, 15)
    rng = np.random.default_rng(1)
    R = rng.normal(size=X.shape)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    bad = ce.agreement_for_label(R, labels, 15)
    assert good.mean() > 0.95
    assert 0.2 < bad.mean() < 0.5  # three balanced clusters -> chance is 1/3
    boot = ce.paired_bootstrap(good, bad, n=200)
    assert boot["ci_low"] > 0 and boot["diff"] > 0.5


def test_unlabelled_rows_are_excluded_not_penalised():
    X, labels = _clustered()
    labels = list(labels)
    labels[0] = frozenset()  # one unlabelled row
    a = ce.agreement_for_label(X, labels, 15)
    assert len(a) == len(labels) - 1


def test_multilabel_any_overlap():
    neigh = np.array([[1, 2]])
    labels = [frozenset(["a", "b"]), frozenset(["b"]), frozenset(["z"])]
    assert ce.label_agreement(neigh, labels)[0] == 0.5


def test_null_is_near_chance():
    X, labels = _clustered()
    neigh = ce.knn_indices(X, 15)
    null = ce.null_agreement(neigh, labels, n=20)
    assert 0.25 < null < 0.42


def test_overlap_is_one_for_identical_and_low_for_random():
    X, _ = _clustered()
    a = ce.knn_indices(X, 15)
    assert ce.neighbourhood_overlap(a, a) == 1.0
    rng = np.random.default_rng(2)
    R = rng.normal(size=X.shape)
    assert ce.neighbourhood_overlap(a, ce.knn_indices(R, 15)) < 0.2


def test_label_sets_from_catalog_columns():
    courses = pd.DataFrame(
        {
            "gers": [["GER:DB-Hum", "WAY-A-II"], [], ["WAY-SI"]],
            "department_code": ["HISTORY", "CS", "CS"],
            "school": ["H&S", "ENGR", "ENGR"],
        }
    )
    ls = ce.label_sets(courses)
    assert ls["breadth (GER:DB)"] == [frozenset(["GER:DB-Hum"]), frozenset(), frozenset()]
    assert ls["WAYS"][2] == frozenset(["WAY-SI"])
    assert ls["department"][1] == frozenset(["CS"])


def test_decision_rule_keeps_incumbent_without_a_clear_win():
    res = {
        "embedding_space": {"inc": {"breadth (GER:DB) k=15": 0.70}, "cand": {"breadth (GER:DB) k=15": 0.71}},
        "paired_vs_incumbent": {"cand": {"diff": 0.01, "ci_low": -0.005, "ci_high": 0.025}},
    }
    d = ce.decide(res, "inc", ["inc", "cand"], "breadth (GER:DB)")
    assert d["winner"] == "inc"
    res["paired_vs_incumbent"]["cand"] = {"diff": 0.03, "ci_low": 0.01, "ci_high": 0.05}
    assert ce.decide(res, "inc", ["inc", "cand"], "breadth (GER:DB)")["winner"] == "cand"
    # step 5: a 2-d collapse beyond the incumbent's seed range sends it back to the incumbent
    res["layout_2d"] = {
        "inc": {"breadth (GER:DB)": {"mean": 0.60, "min": 0.59, "max": 0.61}},
        "cand": {"breadth (GER:DB)": {"mean": 0.50, "min": 0.49, "max": 0.51}},
    }
    assert ce.decide(res, "inc", ["inc", "cand"], "breadth (GER:DB)")["winner"] == "inc"


def test_chunked_trustworthiness_matches_sklearn():
    from sklearn.manifold import trustworthiness as sk_trust

    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 32))
    Y = X[:, :2] + 0.3 * rng.normal(size=(300, 2))
    ours = ce.trustworthiness(X, Y, 15, chunk=64)
    ref = sk_trust(X, Y, n_neighbors=15, metric="cosine")
    assert abs(ours - ref) < 1e-9
