"""Model selection along the Pareto front via extrapolation-shaped validation.

Candidates are scored by cross-validation on folds that refit on the inner
part of the training domain and validate on its outer ring, mimicking the test
scenario without touching test data.  The winner is the *simplest* candidate
whose mean log CV error is within one standard error of the best, which is
robust to CV noise on the training-loss plateaus these fronts tend to have.
"""
from __future__ import annotations

import numpy as np

from .search import Candidate, fit_skeleton, sparsify

_EPS = 1e-12


def extrapolation_folds(X: np.ndarray, quantiles=(0.5, 0.6, 0.7, 0.8)):
    """Folds (inner_mask, outer_mask) at several extrapolation horizons: points
    are ranked by sup-norm distance from the domain center, the inner ``q``
    fraction forms the fit set, the outer ring the validation set.
    """
    center = (X.max(axis=0) + X.min(axis=0)) / 2.0
    r = np.max(np.abs(X - center), axis=1)
    folds = []
    for q in quantiles:
        thr = np.quantile(r, q)
        inner = r <= thr
        outer = ~inner
        if inner.sum() >= 8 and outer.sum() >= 4:
            folds.append((inner, outer))
    return folds


def _dilated_grid(X: np.ndarray, dilation: float, n: int = 256) -> np.ndarray:
    """Deterministic grid over the training box scaled by ``dilation``."""
    center = (X.max(axis=0) + X.min(axis=0)) / 2.0
    radius = np.max(np.abs(X - center), axis=0) * dilation
    d = X.shape[1]
    if d == 1:
        return (center + np.linspace(-radius[0], radius[0], n).reshape(-1, 1))
    m = int(np.sqrt(n))
    axes = [np.linspace(c - r, c + r, m) for c, r in zip(center, radius)]
    mesh = np.meshgrid(*axes)
    return np.stack([v.ravel() for v in mesh], axis=1)


def growth_filter(front: list[Candidate], X: np.ndarray, y: np.ndarray,
                  dilation: float = 1.5, max_ratio: float = 1e6):
    """Drop candidates that blow up just outside the training box.

    Cross-validation cannot see this: even extrapolation-shaped folds validate
    only up to the edge of the data, so a core like exp(exp(x)) -- unremarkable
    on the box, infinite shortly beyond it -- survives every in-domain
    criterion.  Candidates are evaluated on the box dilated by ``dilation`` and
    dropped if not finite there or beyond ``max_ratio`` times the scale of y;
    both thresholds are loose enough for legitimate fast growth.  If all are
    dropped, the front is kept unchanged.
    """
    G = _dilated_grid(X, dilation)
    scale = max(float(np.max(np.abs(y))), 1e-12)
    keep = []
    for c in front:
        with np.errstate(all="ignore"):
            pred = c.predict(G)
        if np.all(np.isfinite(pred)) and float(np.max(np.abs(pred))) <= max_ratio * scale:
            keep.append(c)
    return keep if keep else front


def interior_folds(X: np.ndarray, n_folds: int = 3, seed: int = 0):
    """Standard random K-fold splits (used by the ablation only)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    parts = np.array_split(idx, n_folds)
    folds = []
    for p in parts:
        outer = np.zeros(len(X), dtype=bool)
        outer[p] = True
        folds.append((~outer, outer))
    return folds


def _cv_matrix(front: list[Candidate], X, y, folds, rng, n_restarts=1):
    """cv[i, f] = validation MSE of candidate i refitted on fold f's inner set."""
    cv = np.full((len(front), len(folds)), np.inf)
    for i, cand in enumerate(front):
        for f, (inner, outer) in enumerate(folds):
            refit = fit_skeleton(cand.expr, X[inner], y[inner], cand.poly_degree,
                                 rng, n_restarts=n_restarts, warm_start=cand.theta,
                                 max_nfev=100)
            if refit is None:
                continue
            refit = sparsify(refit, X[inner], y[inner])
            with np.errstate(all="ignore"):
                pred = refit.predict(X[outer])
            err = pred - y[outer]
            err = np.where(np.isfinite(err), np.clip(err, -1e6, 1e6), 1e6)
            cv[i, f] = float(np.mean(err ** 2))
    return cv


def select(front: list[Candidate], X, y, folds=None, seed: int = 0,
           guard: bool = True) -> tuple[Candidate, dict]:
    """Pick the final expression from the Pareto front (1-SE rule).

    ``info`` reports the winner's CV score even for a single-element front, so
    that whole search configurations can be compared by it.  ``guard`` applies
    the out-of-domain growth filter beforehand.
    """
    rng = np.random.default_rng(seed)
    n_before = len(front)
    if guard:
        front = growth_filter(front, X, y)
    if folds is None:
        folds = extrapolation_folds(X)
    if not folds:
        best = min(front, key=lambda c: c.mse)
        return best, {}
    cv = _cv_matrix(front, X, y, folds, rng)
    logcv = np.log(cv + _EPS)
    mean_log = np.where(np.all(np.isfinite(logcv), axis=1),
                        logcv.mean(axis=1), np.inf)

    j_star = int(np.argmin(mean_log))
    se = float(np.std(logcv[j_star], ddof=1) / np.sqrt(len(folds)))
    threshold = mean_log[j_star] + se

    # simplest candidate within 1 SE
    order = sorted(range(len(front)),
                   key=lambda i: (front[i].complexity, front[i].core_complexity))
    chosen_i = j_star
    for i in order:
        if mean_log[i] <= threshold:
            chosen_i = i
            break
    info = {"cv": cv, "mean_log": mean_log, "se": se, "threshold": threshold,
            "best_index": j_star, "chosen_index": chosen_i,
            "n_vetoed": n_before - len(front),
            "cv_score": float(mean_log[chosen_i])}
    return front[chosen_i], info
