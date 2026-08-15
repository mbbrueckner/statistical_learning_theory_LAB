"""Evaluation metrics for the extrapolation task."""
from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.where(np.isfinite(pred), np.clip(pred, -1e9, 1e9), 1e9)
    return float(np.mean((pred - truth) ** 2))


def nmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """nmse = 1 corresponds to predicting the mean; comparable across output scales."""
    var = float(np.var(truth))
    return mse(pred, truth) / max(var, 1e-12)


def functional_recovery(predict, dataset, tol: float = 1e-6) -> bool:
    """True if the model matches the ground truth to relative RMSE ``tol`` on a
    dense grid over the full box [-b, b]^d -- functional, not symbolic, identity."""
    from .generator import _dense_grid

    grid = _dense_grid(dataset.b, dataset.d, n=4096)
    truth = dataset.truth(grid)
    with np.errstate(all="ignore"):
        pred = predict(grid)
    if not np.all(np.isfinite(pred)):
        return False
    scale = max(float(np.std(truth)), 1e-12)
    return float(np.sqrt(np.mean((pred - truth) ** 2))) / scale < tol
