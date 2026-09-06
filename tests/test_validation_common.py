import numpy as np

import validation_common as vc


def test_course_number_and_level():
    assert vc.course_number("HISTORY 158C") == 158
    assert vc.course_number("CS 106A") == 106
    assert vc.course_number("MATH 51") == 51
    assert vc.course_number("EE 101A") == 101
    assert [vc.level_of(k) for k in (1, 99, 100, 199, 200, 299, 300, 999)] == [
        "1-99", "1-99", "100-199", "100-199", "200-299", "200-299", "300+", "300+"
    ]  # fmt: skip


def test_share_same_counts_neighbours_with_the_same_label():
    labels = np.array(["a", "a", "b", "b"])
    neigh = np.array([[1, 2], [0, 3], [3, 0], [2, 1]])
    assert vc.share_same(neigh, labels).tolist() == [0.5, 0.5, 0.5, 0.5]
    assert vc.share_same(np.array([[1], [0], [3], [2]]), labels).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_department_centroids_are_unit_norm_and_respect_mask_and_min_size():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    dept = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 2)
    codes, C, counts = vc.department_centroids(X, dept, min_size=3)
    assert codes == ["a", "b"] and counts.tolist() == [5, 3]
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0)
    expected = X[:5].mean(axis=0)
    assert np.allclose(C[0], expected / np.linalg.norm(expected))
    mask = np.array([True] * 3 + [False] * 7)
    codes, C, counts = vc.department_centroids(X, dept, mask=mask)
    assert codes == ["a"] and counts.tolist() == [3]


def test_within_group_contrasts_recovers_a_planted_effect_that_the_raw_means_hide():
    rng = np.random.default_rng(1)
    # Two departments with different baselines; the low-baseline one is mostly level B, so raw means show a
    # negative B effect while the true within-department effect is +0.2.
    group = np.array(["hi"] * 400 + ["lo"] * 400)
    level = np.array(["A"] * 300 + ["B"] * 100 + ["A"] * 100 + ["B"] * 300)
    base = np.where(group == "hi", 0.7, 0.2)
    y = base + 0.2 * (level == "B") + 0.05 * rng.normal(size=800)
    raw_diff = y[level == "B"].mean() - y[level == "A"].mean()
    assert raw_diff < 0
    res = vc.within_group_contrasts(y, group, level, ["A", "B"], n_boot=200)
    est = res["contrasts"]["B"]
    assert abs(est["estimate"] - 0.2) < 0.02
    assert est["ci_low"] < 0.2 < est["ci_high"]
    assert res["n_groups"] == 2 and res["baseline"] == "A"


def test_bootstrap_mean_interval_contains_the_mean_and_drops_nan():
    vals = np.array([1.0, 2.0, 3.0, np.nan, 4.0])
    res = vc.bootstrap_mean(vals, n_boot=200)
    assert res["n"] == 4 and res["mean"] == 2.5
    assert res["ci_low"] <= 2.5 <= res["ci_high"]
    assert vc.bootstrap_mean(np.array([np.nan]))["n"] == 0
