"""Step 4 of docs/plan.md, check 3 (lit_review 4.3.5): a linear probe for school and department, embeddings versus tf-idf.

    OMP_NUM_THREADS=1 uv run python experiments/linear_probe.py [--source stanford] [--quick]

Open Syllabus classifies syllabi into 70 CIP fields from bag-of-words at 84%. The comparable number here, and the
supervised complement to the preregistered kNN agreement: how much of the org chart a linear classifier recovers from
content alone, from the unit-norm Qwen3-0.6B embeddings and from tf-idf over the same embedded text (word unigrams and
bigrams, min_df 2, sublinear tf, l2 rows). Classifier: linear SVM (LinearSVC, one-vs-rest, squared hinge), C chosen from
{0.1, 1, 10} by 3-fold inner cross-validation on each training fold; 5-fold stratified outer cross-validation, seed 0.
Targets: school (9 classes) and department; department classes are those with at least MIN_DEPARTMENT_COURSES courses,
and the same rows are used for every target and feature set so the numbers compare. Baselines: majority class and the
expected accuracy of a frequency-matched random guess. For department, the share of errors that land in the correct
school says whether the misses are org-chart-near.

Writes data/<source>/validation_probe.json, validation_probe_predictions.parquet (per course, cross-validated
predictions) and validation_probe_departments.csv (per department recall); pipeline/07_record.py copies them.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
from io_utils import write_parquet_safely  # noqa: E402
from validation_common import RNG_SEED, load_map, run_record, write_json  # noqa: E402

MIN_DEPARTMENT_COURSES = 10
C_GRID = [0.1, 1.0, 10.0]
OUTER_FOLDS = 5
INNER_FOLDS = 3
FEATURES = ("embedding", "tfidf")
TARGETS = ("school", "department")


def make_model(features: str, c_grid: list[float]):
    svc = LinearSVC(dual=True, max_iter=10_000)
    if features == "tfidf":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        return make_pipeline(vec, svc), {"linearsvc__C": c_grid}
    return svc, {"C": c_grid}


def fit_fold(
    features: str, F, y: np.ndarray, train: np.ndarray, test: np.ndarray, c_grid, inner_folds: int, seed: int
) -> dict:
    model, grid = make_model(features, c_grid)
    inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(model, grid, cv=inner, n_jobs=1, refit=True)
    t0 = time.time()
    search.fit(F[train], y[train])
    return {
        "test": test,
        "pred": search.predict(F[test]),
        "C": float(list(search.best_params_.values())[0]),
        "inner_accuracy": {str(c): float(s) for c, s in zip(c_grid, search.cv_results_["mean_test_score"])},
        "seconds": round(time.time() - t0),
    }


def probe(
    features: str,
    F,
    y: np.ndarray,
    outer_folds: int = OUTER_FOLDS,
    inner_folds: int = INNER_FOLDS,
    c_grid=C_GRID,
    seed: int = RNG_SEED,
    n_jobs: int = OUTER_FOLDS,
) -> tuple[np.ndarray, list[dict]]:
    """Cross-validated predictions for every row and the per-fold records."""
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    folds = Parallel(n_jobs=n_jobs)(
        delayed(fit_fold)(features, F, y, tr, te, c_grid, inner_folds, seed) for tr, te in outer.split(F, y)
    )
    pred = np.empty(len(y), dtype=object)
    for f in folds:
        pred[f["test"]] = f["pred"]
    return pred, folds


def scores(y: np.ndarray, pred: np.ndarray, folds: list[dict]) -> dict:
    per_fold = [float((pred[f["test"]] == y[f["test"]]).mean()) for f in folds]
    return {
        "accuracy": float((pred == y).mean()),
        "accuracy_by_fold": per_fold,
        "accuracy_fold_sd": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else None,
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "chosen_C_by_fold": [f["C"] for f in folds],
        "inner_accuracy_by_fold": [f["inner_accuracy"] for f in folds],
        "seconds_by_fold": [f["seconds"] for f in folds],
    }


def baselines(y: np.ndarray) -> dict:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return {
        "n_classes": int(len(counts)),
        "majority_class": float(p.max()),
        "frequency_matched_guess": float((p**2).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=config.DEFAULT_SOURCE)
    ap.add_argument("--quick", action="store_true", help="two outer folds, C fixed at 1, no inner search")
    args = ap.parse_args()
    outer_folds, c_grid = (2, [1.0]) if args.quick else (OUTER_FOLDS, C_GRID)
    t0 = time.time()
    corpus, X, _ = load_map(args.source)
    dept_counts = corpus["department_code"].value_counts()
    rows = corpus["department_code"].isin(dept_counts[dept_counts >= MIN_DEPARTMENT_COURSES].index).to_numpy()
    idx = np.flatnonzero(rows)
    texts = corpus["embed_text"].to_numpy(dtype=object)[idx]
    F = {"embedding": X[idx], "tfidf": texts}
    Y = {"school": corpus["school"].to_numpy()[idx], "department": corpus["department_code"].to_numpy()[idx]}
    school_of = corpus.groupby("department_code")["school"].first().to_dict()
    print(
        f"{len(idx)} of {len(corpus)} courses in departments with >= {MIN_DEPARTMENT_COURSES} courses ({len(np.unique(Y['department']))} departments, {len(np.unique(Y['school']))} schools)"
    )

    preds = pd.DataFrame(
        {"course_id": corpus["course_id"].to_numpy()[idx], "school": Y["school"], "department": Y["department"]}
    )
    result = {}
    for target in TARGETS:
        result[target] = {"baselines": baselines(Y[target])}
        for features in FEATURES:
            t1 = time.time()
            pred, folds = probe(
                features, F[features], Y[target], outer_folds=outer_folds, c_grid=c_grid, n_jobs=outer_folds
            )
            preds[f"pred_{target}_{features}"] = pred
            result[target][features] = scores(Y[target], pred, folds) | {"seconds": round(time.time() - t1)}
            if target == "department":
                wrong = pred != Y[target]
                same_school = np.array([school_of[p] == school_of[t] for p, t in zip(pred[wrong], Y[target][wrong])])
                result[target][features]["errors_in_correct_school"] = (
                    float(same_school.mean()) if wrong.any() else None
                )
            r = result[target][features]
            print(
                f"  {target:10} {features:9} accuracy {r['accuracy']:.3f} (folds sd {r['accuracy_fold_sd'] if r['accuracy_fold_sd'] is None else round(r['accuracy_fold_sd'], 3)}), macro-F1 {r['macro_f1']:.3f}, C {r['chosen_C_by_fold']}, {r['seconds']}s"
                + (f", errors in correct school {r['errors_in_correct_school']:.3f}" if target == "department" else "")
            )

    per_dept = (
        preds.assign(
            hit_embedding=preds["pred_department_embedding"] == preds["department"],
            hit_tfidf=preds["pred_department_tfidf"] == preds["department"],
        )
        .groupby("department")
        .agg(
            school=("school", "first"),
            n=("course_id", "size"),
            recall_embedding=("hit_embedding", "mean"),
            recall_tfidf=("hit_tfidf", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    base = config.source_paths(args.source)["base"]
    per_dept.to_csv(base / "validation_probe_departments.csv", index=False)
    write_parquet_safely(preds, base / "validation_probe_predictions.parquet")
    out = run_record(
        args.source,
        quick=args.quick,
        min_department_courses=MIN_DEPARTMENT_COURSES,
        outer_folds=outer_folds,
        inner_folds=INNER_FOLDS,
        c_grid=c_grid,
        seed=RNG_SEED,
        classifier="LinearSVC(dual=True, max_iter=10000), one-vs-rest, squared hinge",
        tfidf="TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), fit on each training fold, over embed_text",
        embedding="unit-norm Qwen3-Embedding-0.6B vectors of embed_text, as in the map",
        definitions={
            "accuracy": "pooled over the outer test folds (every course predicted once)",
            "macro_f1": "unweighted mean of per-class F1 over the pooled predictions",
            "frequency_matched_guess": "sum of squared class frequencies: expected accuracy of guessing by class frequency",
            "errors_in_correct_school": "of the wrong department predictions, the share whose predicted department is in the true school",
        },
        n_courses=int(len(idx)),
        n_courses_total=int(len(corpus)),
        results=result,
        seconds=round(time.time() - t0),
    )
    write_json(base / "validation_probe.json", out)
    print(
        f"wrote validation_probe.json, validation_probe_predictions.parquet, validation_probe_departments.csv in {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
