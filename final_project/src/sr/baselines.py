"""Baseline regressors: polynomial ridge, random forest, MLP.

Hyperparameters are selected with the same extrapolation-shaped validation
folds as the symbolic regressor, so that every method gets the same chance
to prepare for extrapolation.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .selection import extrapolation_folds


def _tune(make_model, param_grid, X, y, folds):
    """Return the model (fitted on all data) whose params minimize mean
    validation MSE over the folds."""
    best_params, best_loss = None, np.inf
    for params in param_grid:
        losses = []
        for inner, outer in folds:
            model = make_model(**params)
            model.fit(X[inner], y[inner])
            pred = model.predict(X[outer])
            pred = np.where(np.isfinite(pred), np.clip(pred, -1e6, 1e6), 1e6)
            losses.append(np.mean((pred - y[outer]) ** 2))
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_loss, best_params = loss, params
    model = make_model(**best_params)
    model.fit(X, y)
    return model, best_params


def fit_polynomial(X, y, folds=None, seed: int = 0):
    folds = folds if folds is not None else extrapolation_folds(X)

    def make(degree, alpha):
        return make_pipeline(PolynomialFeatures(degree), StandardScaler(),
                             Ridge(alpha=alpha))

    grid = [{"degree": d, "alpha": a}
            for d in range(1, 9) for a in (1e-4, 1e-2, 1.0)]
    return _tune(make, grid, X, y, folds)


def fit_random_forest(X, y, folds=None, seed: int = 0):
    folds = folds if folds is not None else extrapolation_folds(X)

    def make(max_depth):
        return RandomForestRegressor(n_estimators=200, max_depth=max_depth,
                                     random_state=seed, n_jobs=1)

    grid = [{"max_depth": m} for m in (4, 8, None)]
    return _tune(make, grid, X, y, folds)


def fit_mlp(X, y, folds=None, seed: int = 0):
    folds = folds if folds is not None else extrapolation_folds(X)

    def make(hidden, alpha):
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=hidden, alpha=alpha,
                         max_iter=4000, random_state=seed, tol=1e-6),
        )

    grid = [{"hidden": h, "alpha": a}
            for h in ((64, 64), (256,)) for a in (1e-4, 1e-2)]
    return _tune(make, grid, X, y, folds)


BASELINES = {
    "polynomial": fit_polynomial,
    "random_forest": fit_random_forest,
    "mlp": fit_mlp,
}
