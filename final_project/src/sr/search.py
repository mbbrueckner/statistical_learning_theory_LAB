"""Systematic complexity-ordered search over expression skeletons.

The regression model is

    f(x) = alpha * u_theta(x) + sum_j beta_j m_j(x) ,

a searched skeleton (the *core*) plus fixed monomials up to ``poly_degree``
(the *polynomial background*).  For a given core, (alpha, beta) are a
closed-form least-squares solution and are projected out, leaving only theta
to optimize; the background also keeps a core like x*sin(x) scoring well when
an additive polynomial trend it does not explain dominates the data.

Ordered by core complexity: phase 1 enumerates all canonical skeletons up to
``k_exhaustive``, phase 2 grows larger ones with a diversity-aware beam.  The
result is a Pareto front, the best model per core complexity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .expressions import Expr, enumerate_skeletons, evaluate, expand, to_string

_BIG = 1e6          # residual value substituted for invalid predictions
_MAX_PARAMS = 6     # skeletons with more free constants are skipped


# ---------------------------------------------------------------------------
# Polynomial background
# ---------------------------------------------------------------------------

def poly_features(X: np.ndarray, degree: int) -> np.ndarray:
    """Monomial features: intercept, x_j^p for p<=degree, and (d=2) x_0*x_1."""
    cols = [np.ones(len(X))]
    names = ["1"]
    for j in range(X.shape[1]):
        for p in range(1, degree + 1):
            cols.append(X[:, j] ** p)
            names.append(f"x{j}^{p}" if p > 1 else f"x{j}")
    if X.shape[1] == 2 and degree >= 2:
        cols.append(X[:, 0] * X[:, 1])
        names.append("x0*x1")
    M = np.stack(cols, axis=1)
    return M, names


def _solve_linear(u: np.ndarray | None, M: np.ndarray, y: np.ndarray):
    """Closed-form (alpha, beta) for y ~ alpha*u + M beta.  u may be None."""
    P = M if u is None else np.concatenate([u[:, None], M], axis=1)
    coef, *_ = np.linalg.lstsq(P, y, rcond=None)
    resid = y - P @ coef
    if u is None:
        return 0.0, coef, resid
    return float(coef[0]), coef[1:], resid


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Candidate:
    """Fitted model  alpha * u_theta(x) + M(x) beta."""
    expr: Expr
    theta: np.ndarray
    alpha: float
    beta: np.ndarray
    poly_degree: int
    mse: float
    poly_names: list = field(default_factory=list)

    @property
    def core_complexity(self) -> int:
        return self.expr.complexity()

    @property
    def complexity(self) -> int:
        """Total complexity: core operator nodes + active background terms."""
        nnz = int(np.sum(np.abs(self.beta[1:]) > 0)) if len(self.beta) else 0
        return self.core_complexity + nnz

    def predict(self, X: np.ndarray) -> np.ndarray:
        M, _ = poly_features(X, self.poly_degree)
        with np.errstate(all="ignore"):
            u = evaluate(self.expr, X, self.theta)
        return self.alpha * u + M @ self.beta

    def to_string(self, digits: int = 3) -> str:
        core = to_string(self.expr, self.theta, digits)
        parts = []
        if abs(self.alpha) > 0:
            parts.append(f"{self.alpha:.{digits}g}*{core}")
        for b, name in zip(self.beta, self.poly_names):
            if abs(b) > 0:
                parts.append(f"{b:.{digits}g}" if name == "1" else f"{b:.{digits}g}*{name}")
        return " + ".join(parts) if parts else "0"

    def to_sympy(self):
        import sympy as sp
        from .expressions import to_sympy as expr_to_sympy

        f = sp.Float(self.alpha) * expr_to_sympy(self.expr, self.theta)
        x = [sp.Symbol(f"x{j}") for j in range(2)]
        for b, name in zip(self.beta, self.poly_names):
            if abs(b) > 0:
                f = f + sp.Float(b) * sp.sympify(name.replace("^", "**"),
                                                 locals={"x0": x[0], "x1": x[1]})
        return sp.simplify(sp.nsimplify(f, rational=False, tolerance=1e-10))

    def __repr__(self):
        return f"[k={self.core_complexity}+{self.complexity - self.core_complexity}, mse={self.mse:.3g}] {self.to_string()}"


# ---------------------------------------------------------------------------
# Fitting a single skeleton (variable projection)
# ---------------------------------------------------------------------------

def fit_skeleton(expr: Expr, X: np.ndarray, y: np.ndarray, poly_degree: int,
                 rng: np.random.Generator, n_restarts: int = 3,
                 warm_start: np.ndarray | None = None,
                 max_nfev: int = 60) -> Candidate | None:
    """Fit theta by multi-start nonlinear least squares; (alpha, beta) are
    solved in closed form inside the residual (variable projection)."""
    M, names = poly_features(X, poly_degree)
    p = expr.n_params()

    def full_residual(theta):
        with np.errstate(all="ignore"):
            u = evaluate(expr, X, theta)
        if not np.all(np.isfinite(u)):
            return None, None, None, np.full(len(y), _BIG)
        u = np.clip(u, -_BIG, _BIG)
        alpha, beta, resid = _solve_linear(u, M, y)
        return u, alpha, beta, resid

    if p == 0:
        u, alpha, beta, resid = full_residual(np.empty(0))
        if u is None:
            return None
        return Candidate(expr, np.empty(0), alpha, beta, poly_degree,
                         float(np.mean(resid ** 2)), names)
    if p > _MAX_PARAMS:
        return None

    def residual(theta):
        return full_residual(theta)[3]

    starts = []
    if warm_start is not None and len(warm_start) == p:
        starts.append(np.asarray(warm_start, dtype=float))
    starts.append(np.ones(p))
    while len(starts) < n_restarts + (warm_start is not None):
        starts.append(rng.normal(0.0, 2.0, size=p))

    best_theta, best_mse = None, np.inf
    for theta0 in starts:
        try:
            res = least_squares(residual, theta0, method="trf",
                                max_nfev=max_nfev, ftol=1e-10, xtol=1e-10)
        except Exception:
            continue
        m = float(np.mean(res.fun ** 2))
        if m < best_mse:
            best_mse, best_theta = m, res.x
    if best_theta is None:
        return None
    u, alpha, beta, resid = full_residual(best_theta)
    if u is None:
        return None
    return Candidate(expr, best_theta, alpha, beta, poly_degree,
                     float(np.mean(resid ** 2)), names)


def sparsify(cand: Candidate, X: np.ndarray, y: np.ndarray,
             rel_tol: float = 1e-4) -> Candidate:
    """Drop background terms whose contribution is negligible, then refit the
    linear part.  A term is negligible if |beta_j| * std(m_j) < rel_tol * std(y)."""
    M, names = poly_features(X, cand.poly_degree)
    with np.errstate(all="ignore"):
        u = evaluate(cand.expr, X, cand.theta)
    u = np.clip(u, -_BIG, _BIG)
    scale = max(float(np.std(y)), 1e-12)
    keep = np.ones(len(cand.beta), dtype=bool)
    for j in range(len(cand.beta)):
        contrib = np.abs(cand.beta[j]) * max(float(np.std(M[:, j])), 1.0 if j == 0 else 0.0)
        if j == 0:
            contrib = abs(cand.beta[j])
        if contrib < rel_tol * scale:
            keep[j] = False
    # also test dropping the core if it contributes nothing
    alpha = cand.alpha
    if abs(alpha) * max(float(np.std(u)), 1e-12) < rel_tol * scale:
        alpha = 0.0
    P_cols, col_ids = [], []
    if alpha != 0.0:
        P_cols.append(u)
    for j in range(len(cand.beta)):
        if keep[j]:
            P_cols.append(M[:, j])
            col_ids.append(j)
    if not P_cols:
        beta = np.zeros(len(cand.beta))
        return Candidate(cand.expr, cand.theta, 0.0, beta, cand.poly_degree,
                         float(np.mean(y ** 2)), names)
    P = np.stack(P_cols, axis=1)
    coef, *_ = np.linalg.lstsq(P, y, rcond=None)
    resid = y - P @ coef
    beta = np.zeros(len(cand.beta))
    off = 0
    new_alpha = 0.0
    if alpha != 0.0:
        new_alpha = float(coef[0])
        off = 1
    for i, j in enumerate(col_ids):
        beta[j] = coef[off + i]
    return Candidate(cand.expr, cand.theta, new_alpha, beta, cand.poly_degree,
                     float(np.mean(resid ** 2)), names)


# ---------------------------------------------------------------------------
# Semantic deduplication by evaluation fingerprints
# ---------------------------------------------------------------------------
# Canonicalization (expressions.py) removes only duplicates provable by
# rewriting.  Since alpha and beta are free, many more cores are numerically
# interchangeable; we detect those by hashing a skeleton's normalized values on
# fixed probe points for several generic parameter draws.
#
#   "affine"      -- normalize mean/scale/sign: merges u, -u, 2u+3
#   "background"  -- also project out the monomials: merges u and u + p(x)
#
# "background" is sharper but riskier: a core redundant on its own may still be
# a necessary intermediate, so both modes are treated as a hyperparameter.

_FP_DECIMALS = 6


def _normalize_fp(v: np.ndarray) -> np.ndarray | None:
    """Scale/offset/sign-canonical form of an evaluation vector."""
    s = float(np.std(v))
    if s < 1e-12:
        return None                       # constant: absorbed by the intercept
    z = (v - float(np.mean(v))) / s
    i = int(np.argmax(np.abs(z)))
    if z[i] < 0:
        z = -z                            # alpha absorbs the sign
    return np.round(z, _FP_DECIMALS)


class Fingerprinter:
    """Maps a skeleton to a hashable key of its function family on the data."""

    def __init__(self, X: np.ndarray, poly_degree: int, mode: str = "affine",
                 n_probe: int = 24, n_theta: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed + 9973)
        idx = rng.choice(len(X), size=min(n_probe, len(X)), replace=False)
        self.Xp = X[idx]
        self.mode = mode
        self.thetas = rng.normal(0.0, 1.5, size=(n_theta, _MAX_PARAMS))
        if mode == "background":
            M, _ = poly_features(self.Xp, poly_degree)
            self.Q, _ = np.linalg.qr(M)
        else:
            self.Q = None

    def __call__(self, expr: Expr):
        p = expr.n_params()
        if p > _MAX_PARAMS:
            return None                   # rejected by the fitter anyway
        thetas = self.thetas[:1] if p == 0 else self.thetas
        blocks = []
        for t in thetas:
            with np.errstate(all="ignore"):
                u = evaluate(expr, self.Xp, t[:p])
            if not np.all(np.isfinite(u)):
                return None               # never merge numerically risky cores
            if self.Q is not None:
                u = u - self.Q @ (self.Q.T @ u)
            z = _normalize_fp(u)
            if z is None:
                return b"__const__"       # nothing beyond the background
            blocks.append(z)
        return np.concatenate(blocks).tobytes()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _root_op(expr: Expr) -> str:
    return getattr(expr, "op", "leaf")


def _pick_parents(level: list[Candidate], beam_width: int, n_random: int,
                  rng: np.random.Generator) -> list[Candidate]:
    """Diversity-aware beam: round-robin over root operators by fit quality,
    plus a few random survivors to escape greedy basins."""
    ranked = sorted(level, key=lambda c: c.mse)
    groups: dict[str, list[Candidate]] = {}
    for c in ranked:
        groups.setdefault(_root_op(c.expr), []).append(c)
    parents, depth = [], 0
    while len(parents) < beam_width and any(depth < len(g) for g in groups.values()):
        for op in sorted(groups):
            g = groups[op]
            if depth < len(g) and len(parents) < beam_width:
                parents.append(g[depth])
        depth += 1
    chosen_ids = {id(c) for c in parents}
    rest = [c for c in ranked if id(c) not in chosen_ids]
    if rest and n_random > 0:
        extra = rng.choice(len(rest), size=min(n_random, len(rest)), replace=False)
        parents.extend(rest[i] for i in extra)
    return parents


def pareto_search(X: np.ndarray, y: np.ndarray, unary: tuple, binary: tuple,
                  poly_degree: int = 2, k_exhaustive: int = 3, k_max: int = 7,
                  beam_width: int = 15, n_random_parents: int = 5,
                  n_restarts: int = 3, seed: int = 0, mse_tol: float = 1e-10,
                  n_fit_subsample: int = 64, dedup: str = "none",
                  stats: dict | None = None, verbose: bool = False) -> list[Candidate]:
    """Return the Pareto front of fitted models (best per core complexity).

    During the search, skeletons are fitted on a random subsample of at most
    ``n_fit_subsample`` training points; the front is refitted on the full
    training set and sparsified before it is returned.  With ``dedup`` set to
    "affine" or "background", a skeleton whose fingerprint was seen before is
    skipped without fitting.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    memo: dict = {}
    y_var = max(float(np.var(y)), 1e-12)

    if len(X) > n_fit_subsample:
        sub = rng.choice(len(X), size=n_fit_subsample, replace=False)
        Xs, ys = X[sub], y[sub]
    else:
        Xs, ys = X, y

    fingerprint = (Fingerprinter(Xs, poly_degree, mode=dedup, seed=seed)
                   if dedup != "none" else None)
    seen_fp: set = set()
    n_seen = n_skipped = n_fitted = 0

    front: dict[int, Candidate] = {}
    level: list[Candidate] = []

    def consider(expr):
        """Fit a skeleton unless an equivalent one was already fitted."""
        nonlocal n_seen, n_skipped, n_fitted
        n_seen += 1
        if fingerprint is not None:
            fp = fingerprint(expr)
            if fp is not None:
                if fp in seen_fp:
                    n_skipped += 1
                    return None
                seen_fp.add(fp)
        n_fitted += 1
        p = expr.n_params()
        restarts = n_restarts if p >= 3 else min(n_restarts, 2)
        cand = fit_skeleton(expr, Xs, ys, poly_degree, rng, n_restarts=restarts)
        if cand is None or not np.isfinite(cand.mse):
            return None
        k = cand.core_complexity
        if k not in front or cand.mse < front[k].mse:
            front[k] = cand
        return cand

    def best_mse():
        return min((c.mse for c in front.values()), default=np.inf)

    # Phase 1: exhaustive
    for k in range(0, k_exhaustive + 1):
        level = []
        for expr in enumerate_skeletons(k, d, unary, binary, memo):
            cand = consider(expr)
            if cand is not None:
                level.append(cand)
        if verbose:
            print(f"exhaustive k={k}: {len(level)} fitted, best mse={best_mse():.3g}")
        if best_mse() < mse_tol * y_var:
            break

    # Phase 2: beam expansion
    for k in range(k_exhaustive + 1, k_max + 1):
        if best_mse() < mse_tol * y_var or not level:
            break
        parents = _pick_parents(level, beam_width, n_random_parents, rng)
        seen, children = set(), []
        for par in parents:
            for child in expand(par.expr, d, unary, binary):
                key = child.key()
                if key not in seen:
                    seen.add(key)
                    children.append(child)
        level = []
        for expr in children:
            cand = consider(expr)
            if cand is not None:
                level.append(cand)
        if verbose:
            print(f"beam k={k}: {len(children)} skeletons, best mse={best_mse():.3g}")

    if stats is not None:
        stats.update(n_candidates=n_seen, n_fitted=n_fitted, n_skipped=n_skipped)

    for k, cand in list(front.items()):
        refit = fit_skeleton(cand.expr, X, y, poly_degree, rng, n_restarts=1,
                             warm_start=cand.theta, max_nfev=200)
        if refit is not None and np.isfinite(refit.mse):
            front[k] = sparsify(refit, X, y)

    # Pareto-optimal: mse strictly decreasing in k
    result, best = [], np.inf
    for k in sorted(front):
        if front[k].mse < best:
            result.append(front[k])
            best = front[k].mse
    return result
