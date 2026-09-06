import numpy as np

import record_metrics as rm


def _clustered(n_per=100, n_clusters=4, dim=8, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    X = np.concatenate([c + noise * rng.normal(size=(n_per, dim)) for c in centres])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    classes = np.repeat([f"c{i}" for i in range(n_clusters)], n_per)
    return X, classes


def test_knn_preservation_is_one_for_a_faithful_layout_and_low_for_a_shuffled_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    neigh = rm.neighbour_overlap  # sanity: identical sets overlap fully
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert neigh(a, a) == 1.0 and neigh(a, np.array([[7, 8, 9], [4, 0, 0]])) == 1 / 6
    from compare_embeddings import knn_indices

    neigh_hd = knn_indices(X, 5, metric="cosine")
    # Unit vectors in 2-d: euclidean and cosine orderings agree, so the layout X itself preserves every neighbourhood.
    assert rm.knn_preservation(neigh_hd, X) == 1.0
    assert rm.knn_preservation(neigh_hd, X[rng.permutation(200)]) < 0.2


def test_knc_preservation_and_min_size_filter():
    X, classes = _clustered()
    Y = X[:, :2]  # keeps the centroid geometry only loosely; a random 2-d layout keeps none of it
    rng = np.random.default_rng(1)
    R = rng.normal(size=(len(X), 2))
    good = rm.knc_preservation(X, X, classes, k=2)
    assert good == {"value": 1.0, "k": 2, "n_classes": 4}
    assert rm.knc_preservation(X, R, classes, k=2)["value"] <= 1.0
    assert rm.knc_preservation(X, Y, classes, k=2)["n_classes"] == 4
    small = np.array(list(classes) + ["tiny"] * 3)
    Xs = np.vstack([X, X[:3]])
    assert rm.knc_preservation(Xs, Xs, small, k=2, min_size=10)["n_classes"] == 4


def test_cpd_is_one_for_the_same_geometry():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    # Cosine distance on unit vectors is monotone in euclidean distance, so Spearman is exactly 1.
    assert abs(rm.cpd(X, X, n_sample=200, seed=0) - 1.0) < 1e-9
    assert rm.cpd(X, rng.normal(size=(300, 2)), n_sample=200, seed=0) < 0.3


def test_match_layers_by_nearest_count():
    assert rm.match_layers([357, 120, 42, 14], [350, 118, 40, 13]) == [0, 1, 2, 3]
    assert rm.match_layers([357, 120, 42, 14], [350, 40, 13]) == [0, 1, 1, 2]


def test_region_agreement_counts_pairs_and_peers():
    published = np.array([0, 0, 0, 0, 1, 1, -1])
    rerun = np.array([5, 5, 5, -1, 2, 3, 0])
    coassign, stay = rm.region_agreement(published, rerun)
    assert coassign[0] == 0.5  # 3 of 6 pairs share rerun region 5
    assert coassign[1] == 0.0
    np.testing.assert_allclose(stay[:4], [2 / 3, 2 / 3, 2 / 3, 0.0])
    assert stay[4] == 0.0 and np.isnan(stay[6])
    # Present mask: the unclustered member is out of the subsample, so the remaining three agree perfectly.
    present = np.array([True, True, True, False, True, True, True])
    coassign, stay = rm.region_agreement(published, rerun, present)
    assert coassign[0] == 1.0 and np.isnan(stay[3]) and stay[0] == 1.0


def test_region_agreement_needs_two_present_members():
    published = np.array([0, 0, 1])
    rerun = np.array([0, 0, 0])
    coassign, stay = rm.region_agreement(published, rerun)
    assert np.isnan(coassign[1]) and np.isnan(stay[2]) and coassign[0] == 1.0


def test_layer_agreement_reports_ami_and_coverage():
    published = np.array([0, 0, 1, 1, -1, 2])
    identical = rm.layer_agreement(published, published)
    assert identical["ami"] == 1.0 and identical["published_members_unclustered"] == 0.0
    rerun = np.array([3, 3, 4, -1, 1, -1])
    out = rm.layer_agreement(published, rerun)
    assert out["published_members_unclustered"] == 2 / 5
    assert out["share_both_clustered"] == 3 / 6
    assert out["n_clusters"] == 5 and out["unlabelled_share"] == 2 / 6


def test_compare_run_and_summarise_average_over_runs():
    published = [np.array([0, 0, 0, 1, 1, -1])]
    run_a = rm.compare_run(published, [np.array([0, 0, 0, 1, 1, -1])])
    run_b = rm.compare_run(published, [np.array([0, 0, 1, 1, 1, -1])])
    layers, regions, stay = rm.summarise([run_a, run_b], published, 6)
    assert layers[0]["published_n_clusters"] == 2
    assert layers[0]["ami"]["max"] == 1.0
    by_cluster = {r["cluster"]: r for r in regions}
    assert by_cluster[0]["coassign_mean"] == (1.0 + 1 / 3) / 2 and by_cluster[0]["coassign_min"] == 1 / 3
    assert by_cluster[1]["coassign_mean"] == 1.0 and by_cluster[1]["n_runs"] == 2
    np.testing.assert_allclose(stay[0][:3], [(1 + 0.5) / 2, (1 + 0.5) / 2, (1 + 0) / 2])
    assert np.isnan(stay[0][5])
