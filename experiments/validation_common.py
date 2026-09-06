"""Shared pieces for the step-4 validation checks of docs/plan.md (level_agreement.py, crosslisting_distance.py,
linear_probe.py, department_centrality.py): the published map's inputs, course-number levels, department centroids,
bootstrap intervals, a within-department contrast, and the run record every check writes into its JSON.
"""

import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402
from io_utils import atomic_write_text  # noqa: E402

# Stanford Bulletin, "Course Numbering" (read 2026-09-06): 1-99 "primarily for freshmen and sophomores"; 100-199
# "primarily for juniors and seniors; some departments, however, offer courses numbered from 200 through 299 for juniors
# and seniors"; "most courses numbered 200 and above are for graduate students", "no graduate career course is numbered
# below 200", "all courses above 300 are for graduate students". The Bulletin adds that Stanford "does not have a
# standard course catalog numbering system", so the buckets follow the policy's own breakpoints and nothing finer.
COURSE_NUMBERING_SOURCE = "https://bulletin.stanford.edu/academic-polices/courses/course-numbering"
LEVELS = ["1-99", "100-199", "200-299", "300+"]
K_NEIGHBOURS = 15  # the pipeline's n_neighbors and the preregistered k
N_BOOTSTRAP = 1_000
RNG_SEED = 0


def load_map(source: str = config.DEFAULT_SOURCE) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """corpus.parquet with the published embeddings X (unit-norm) and layout Y, both aligned to its rows."""
    paths = config.source_paths(source)
    files = config.keyed_files(paths)
    corpus = pd.read_parquet(paths["corpus"])
    ids = corpus["course_id"].to_numpy()
    emb = np.load(files["npz"], allow_pickle=True)
    lay = np.load(files["umap"], allow_pickle=True)
    assert (emb["course_id"] == ids).all() and (lay["course_id"] == ids).all(), "npz not aligned to corpus.parquet"
    return corpus, emb["embeddings"].astype(np.float64), lay["coords"].astype(np.float64)


def course_number(code: str) -> int:
    """The number in a catalog code: 'HISTORY 158C' -> 158, 'CS 106A' -> 106, 'MATH 51' -> 51."""
    m = re.search(r"\s(\d+)", code)
    if m is None:
        raise ValueError(f"no course number in {code!r}")
    return int(m.group(1))


def level_of(number: int) -> str:
    if number < 100:
        return "1-99"
    if number < 200:
        return "100-199"
    if number < 300:
        return "200-299"
    return "300+"


def share_same(neigh: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per row, the share of its neighbours (indices into `labels`) that carry the row's own label."""
    return (labels[neigh] == labels[:, None]).mean(axis=1)


def department_centroids(
    X: np.ndarray, dept: np.ndarray, mask: np.ndarray | None = None, min_size: int = 1
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Unit-normalised mean embedding per department over the rows in `mask`, for departments with at least
    `min_size` such rows. Returns (department codes, centroids, counts), codes sorted."""
    mask = np.ones(len(X), dtype=bool) if mask is None else mask
    codes, inverse, counts = np.unique(dept[mask], return_inverse=True, return_counts=True)
    sums = np.zeros((len(codes), X.shape[1]))
    np.add.at(sums, inverse, X[mask])
    keep = counts >= min_size
    C = sums[keep] / counts[keep, None]
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return [str(c) for c in codes[keep]], C, counts[keep]


def bootstrap_mean(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> dict:
    """Mean with a 95% percentile bootstrap interval over resampled rows."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "n": int(len(values)),
    }


def fixed_effects_fit(y: np.ndarray, group: np.ndarray, D: np.ndarray) -> np.ndarray:
    """OLS coefficients of y on the columns of D with group fixed effects, by within-group demeaning of y and of every
    column (Frisch-Waugh: identical to including a dummy per group)."""
    g = pd.factorize(group)[0]
    ng = g.max() + 1
    counts = np.bincount(g, minlength=ng).astype(np.float64)
    yd = y - (np.bincount(g, weights=y, minlength=ng) / counts)[g]
    Dd = np.empty(D.shape, dtype=np.float64)
    for j in range(D.shape[1]):
        Dd[:, j] = D[:, j] - (np.bincount(g, weights=D[:, j], minlength=ng) / counts)[g]
    return np.linalg.lstsq(Dd, yd, rcond=None)[0]


def within_group_contrasts(
    y: np.ndarray,
    group: np.ndarray,
    level: np.ndarray,
    levels: list[str],
    n_boot: int = N_BOOTSTRAP,
    seed: int = RNG_SEED,
) -> dict:
    """Each level's mean difference from `levels[0]` after group fixed effects (groups differ in their baseline and in
    their level mix, so raw by-level means confound the two). 95% percentile bootstrap over rows."""
    y = np.asarray(y, dtype=np.float64)
    group = np.asarray(group)
    D = np.stack([(level == lv).astype(np.float64) for lv in levels[1:]], axis=1)
    point = fixed_effects_fit(y, group, D)
    rng = np.random.default_rng(seed)
    boots = np.stack(
        [fixed_effects_fit(y[r], group[r], D[r]) for r in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    )
    return {
        "baseline": levels[0],
        "n_groups": int(len(np.unique(group))),
        "n": int(len(y)),
        "contrasts": {
            lv: {
                "estimate": float(point[j]),
                "ci_low": float(np.percentile(boots[:, j], 2.5)),
                "ci_high": float(np.percentile(boots[:, j], 97.5)),
            }
            for j, lv in enumerate(levels[1:])
        },
    }


def run_record(source: str, **params) -> dict:
    return {
        "source": source,
        "embedding_model": config.EMBED_MODEL_KEY,
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        **params,
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(path: Path, obj: dict) -> None:
    atomic_write_text(Path(path), json.dumps(obj, indent=2, default=_json_default) + "\n")
