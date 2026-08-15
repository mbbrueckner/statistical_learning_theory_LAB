"""Synthetic data generator: random symbolic formulas + extrapolation datasets.

A ground-truth formula is sampled by a "dice roll" over operators:  a random
expression tree with a prescribed number of operator nodes is drawn, its
constant placeholders are instantiated with random values, and degenerate
formulas are rejected.  Training inputs are sampled uniformly from the inner
box [-a, a]^d, test inputs uniformly from the ring [-b, b]^d \\ [-a, a]^d,
so that every test point lies strictly outside the training domain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .expressions import (
    Bin, Const, Expr, Un, Var, evaluate, to_string, OP_SETS,
)


@dataclass
class Dataset:
    expr: Expr                  # ground-truth skeleton
    theta: np.ndarray           # ground-truth constants
    X_train: np.ndarray
    y_train: np.ndarray         # possibly noisy
    X_test: np.ndarray
    y_test: np.ndarray          # noise-free ground truth (evaluation target)
    a: float                    # training half-width
    b: float                    # test half-width
    d: int
    n_ops: int                  # dice-rolled complexity
    op_set: str
    noise: float                # relative noise level sigma_rel
    seed: int
    meta: dict = field(default_factory=dict)

    def truth(self, X):
        return evaluate(self.expr, X, self.theta)

    def __repr__(self):
        return f"Dataset({to_string(self.expr, self.theta)}, d={self.d}, k={self.n_ops})"


def effective_complexity(expr: Expr, theta: np.ndarray) -> int:
    """Operator count of the sympy-simplified formula.  Dice-rolled trees
    often contain collapsing subterms (e.g. x - x), so the sampled operator
    count overstates the true complexity; analyses bin by this value."""
    import sympy as sp

    from .expressions import to_sympy

    try:
        f = sp.simplify(to_sympy(expr, theta))
        return int(sp.count_ops(f))
    except Exception:
        return expr.complexity()


def sample_skeleton(rng: np.random.Generator, n_ops: int, d: int,
                    unary: tuple, binary: tuple, p_unary: float = 0.4) -> Expr:
    """Random expression tree with exactly ``n_ops`` operator nodes."""

    def leaf():
        # favour variables so formulas actually depend on x
        if rng.random() < 0.7:
            return Var(int(rng.integers(d)))
        return Const()

    def grow(budget: int) -> Expr:
        if budget == 0:
            return leaf()
        if budget >= 1 and (rng.random() < p_unary or budget == 1 and not binary):
            return Un(str(rng.choice(unary)), grow(budget - 1))
        if not binary:
            return Un(str(rng.choice(unary)), grow(budget - 1))
        split = int(rng.integers(budget))  # ops for the left subtree
        return Bin(str(rng.choice(binary)), grow(split), grow(budget - 1 - split))

    return grow(n_ops)


def _instantiate_constants(rng: np.random.Generator, n: int) -> np.ndarray:
    """Constants from +-U([0.2, 3]), rounded to 2 decimals."""
    mag = rng.uniform(0.2, 3.0, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    return np.round(mag * sign, 2)


def _dense_grid(b: float, d: int, n: int = 2000) -> np.ndarray:
    if d == 1:
        return np.linspace(-b, b, n).reshape(-1, 1)
    m = int(np.sqrt(n))
    g = np.linspace(-b, b, m)
    xx = np.meshgrid(*([g] * d))
    return np.stack([v.ravel() for v in xx], axis=1)


def _sample_ring(rng, n, a, b, d):
    """Uniform on [-b, b]^d \\ [-a, a]^d by rejection."""
    out = []
    while sum(len(o) for o in out) < n:
        X = rng.uniform(-b, b, size=(4 * n, d))
        mask = np.max(np.abs(X), axis=1) > a
        out.append(X[mask])
    return np.concatenate(out)[:n]


def sample_dataset(seed: int, n_ops: int, d: int = 1, op_set: str = "full",
                   n_train: int = 128, n_test: int = 256,
                   a: float = 5.0, b: float = 8.0, noise: float = 0.0,
                   max_tries: int = 500) -> Dataset:
    """Sample one ground-truth formula and an extrapolation dataset.

    Rejection rules (checked on a dense grid over the *full* box [-b, b]^d):
      R1 invalid values (NaN/Inf), e.g. division blow-ups inside the domain;
      R2 extreme magnitudes  max |f| > 1e4  (dominated by exp overflow);
      R3 numerically constant output  std(f) < 1e-3;
      R4 formula does not depend on any variable;
      R5 effectively affine although n_ops >= 3 (an OLS line fits with
         R^2 > 1 - 1e-9), which would mislabel the complexity level.
    """
    rng = np.random.default_rng(seed)
    ops = OP_SETS[op_set]
    grid = _dense_grid(b, d)

    for attempt in range(max_tries):
        expr = sample_skeleton(rng, n_ops, d, ops["unary"], ops["binary"])
        if not expr.has_var():
            continue
        theta = _instantiate_constants(rng, expr.n_params())
        with np.errstate(all="ignore"):
            f_grid = evaluate(expr, grid, theta)
        if not np.all(np.isfinite(f_grid)):                      # R1
            continue
        if np.max(np.abs(f_grid)) > 1e4:                         # R2
            continue
        if np.std(f_grid) < 1e-3:                                # R3
            continue
        used = set()

        def vars_used(node):
            if isinstance(node, Var):
                used.add(node.j)
            elif isinstance(node, Un):
                vars_used(node.child)
            elif isinstance(node, Bin):
                vars_used(node.left), vars_used(node.right)

        vars_used(expr)
        if len(used) < d:                                        # R4 (all dims used)
            continue
        if n_ops >= 3:                                           # R5
            A = np.hstack([grid, np.ones((len(grid), 1))])
            coef, *_ = np.linalg.lstsq(A, f_grid, rcond=None)
            resid = f_grid - A @ coef
            if np.var(resid) < 1e-9 * np.var(f_grid):
                continue

        X_train = rng.uniform(-a, a, size=(n_train, d))
        X_test = _sample_ring(rng, n_test, a, b, d)
        y_train = evaluate(expr, X_train, theta)
        y_test = evaluate(expr, X_test, theta)
        if not (np.all(np.isfinite(y_train)) and np.all(np.isfinite(y_test))):
            continue
        if noise > 0:
            y_train = y_train + noise * np.std(y_train) * rng.standard_normal(n_train)
        return Dataset(expr, theta, X_train, y_train, X_test, y_test,
                       a, b, d, n_ops, op_set, noise, seed,
                       meta={"attempts": attempt + 1,
                             "effective_ops": effective_complexity(expr, theta)})

    raise RuntimeError(f"could not sample a valid formula (seed={seed}, n_ops={n_ops})")
