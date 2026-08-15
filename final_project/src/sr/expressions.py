"""Expression trees for symbolic regression.

An expression skeleton is a tree whose leaves are either input variables
x_j or constant placeholders c (free parameters).  Internal nodes are
unary or binary operators.  The complexity of a skeleton is its number of
operator nodes.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _safe_log(t):
    return np.log(np.abs(t))


def _safe_sqrt(t):
    return np.sqrt(np.abs(t))


UNARY_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "log": _safe_log,
    "sqrt": _safe_sqrt,
    "square": np.square,
}

BINARY_FUNCS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,      # true division; invalid values handled by caller
}

_BIN_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}

# Named operator sets used by the generator and the experiments.
OP_SETS = {
    "poly": {"unary": ("square",), "binary": ("add", "sub", "mul")},
    "trig": {"unary": ("square", "sin", "cos"), "binary": ("add", "sub", "mul")},
    "full": {
        "unary": ("square", "sin", "cos", "exp", "log", "sqrt"),
        "binary": ("add", "sub", "mul", "div"),
    },
}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class Expr:
    """Base class; nodes are immutable after construction."""

    __slots__ = ()

    # number of operator nodes
    def complexity(self) -> int:
        raise NotImplementedError

    def n_params(self) -> int:
        raise NotImplementedError

    def has_var(self) -> bool:
        raise NotImplementedError

    def key(self) -> str:
        """Canonical string used for structural deduplication."""
        raise NotImplementedError


class Var(Expr):
    __slots__ = ("j",)

    def __init__(self, j: int):
        object.__setattr__(self, "j", j)

    def __setattr__(self, *a):  # immutability
        raise AttributeError("immutable")

    def complexity(self):
        return 0

    def n_params(self):
        return 0

    def has_var(self):
        return True

    def key(self):
        return f"x{self.j}"


class Const(Expr):
    __slots__ = ()

    def complexity(self):
        return 0

    def n_params(self):
        return 1

    def has_var(self):
        return False

    def key(self):
        return "c"


class Un(Expr):
    __slots__ = ("op", "child", "_key", "_cplx", "_np")

    def __init__(self, op: str, child: Expr):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "child", child)
        object.__setattr__(self, "_key", f"{op}({child.key()})")
        object.__setattr__(self, "_cplx", 1 + child.complexity())
        object.__setattr__(self, "_np", child.n_params())

    def __setattr__(self, *a):
        raise AttributeError("immutable")

    def complexity(self):
        return self._cplx

    def n_params(self):
        return self._np

    def has_var(self):
        return self.child.has_var()

    def key(self):
        return self._key


class Bin(Expr):
    __slots__ = ("op", "left", "right", "_key", "_cplx", "_np")

    def __init__(self, op: str, left: Expr, right: Expr):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "_key", f"({left.key()}{_BIN_SYMBOL[op]}{right.key()})")
        object.__setattr__(self, "_cplx", 1 + left.complexity() + right.complexity())
        object.__setattr__(self, "_np", left.n_params() + right.n_params())

    def __setattr__(self, *a):
        raise AttributeError("immutable")

    def complexity(self):
        return self._cplx

    def n_params(self):
        return self._np

    def has_var(self):
        return self.left.has_var() or self.right.has_var()

    def key(self):
        return self._key


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(expr: Expr, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate expr on data X (N, d) with parameters theta.

    Constant placeholders consume entries of theta in depth-first
    left-to-right order.  Returns an array of shape (N,) that may contain
    NaN/Inf (callers must handle invalid values).
    """
    pos = [0]

    def rec(node):
        if isinstance(node, Var):
            return X[:, node.j]
        if isinstance(node, Const):
            v = theta[pos[0]]
            pos[0] += 1
            return np.full(X.shape[0], v)
        with np.errstate(all="ignore"):
            if isinstance(node, Un):
                return UNARY_FUNCS[node.op](rec(node.child))
            return BINARY_FUNCS[node.op](rec(node.left), rec(node.right))

    return np.asarray(rec(expr), dtype=float)


def to_string(expr: Expr, theta=None, digits: int = 3) -> str:
    """Human-readable infix string; instantiates constants if theta given."""
    pos = [0]

    def rec(node):
        if isinstance(node, Var):
            return f"x{node.j}"
        if isinstance(node, Const):
            if theta is None:
                return "c"
            v = theta[pos[0]]
            pos[0] += 1
            return f"{v:.{digits}g}"
        if isinstance(node, Un):
            inner = rec(node.child)
            if node.op == "square":
                return f"({inner})^2"
            return f"{node.op}({inner})"
        return f"({rec(node.left)} {_BIN_SYMBOL[node.op]} {rec(node.right)})"

    return rec(expr)


def to_sympy(expr: Expr, theta=None):
    """Convert to a sympy expression (protected log/sqrt become log|.|, sqrt|.|)."""
    import sympy as sp

    pos = [0]

    def rec(node):
        if isinstance(node, Var):
            return sp.Symbol(f"x{node.j}")
        if isinstance(node, Const):
            if theta is None:
                sym = sp.Symbol(f"c{pos[0]}")
                pos[0] += 1
                return sym
            v = theta[pos[0]]
            pos[0] += 1
            return sp.Float(v)
        if isinstance(node, Un):
            t = rec(node.child)
            return {
                "sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
                "log": lambda u: sp.log(sp.Abs(u)),
                "sqrt": lambda u: sp.sqrt(sp.Abs(u)),
                "square": lambda u: u ** 2,
            }[node.op](t)
        left, right = rec(node.left), rec(node.right)
        return {
            "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
        }[node.op](left, right)

    return rec(expr)


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


_FORBIDDEN_UNARY_PAIRS = {
    ("exp", "log"), ("log", "exp"), ("square", "sqrt"),
}


def _flatten(op: str, node: Expr, out: list):
    """Flatten an associative add/mul chain into its term list."""
    if isinstance(node, Bin) and node.op == op:
        _flatten(op, node.left, out)
        _flatten(op, node.right, out)
    else:
        out.append(node)



def canonicalize(node: Expr) -> Expr | None:
    """
    Maps a raw tree to a canonical representative of its semantic equivalence class,
    or returns None if the tree # is redundant

    Note: under the protected semantics log|t|, sqrt|t| several compositions
    collapse to the absolute value:  exp(log(t)) = square(sqrt(t)) =
    sqrt(square(t)) = |t|.  We keep sqrt(square(t)) as the canonical
    representation of |t| and forbid the aliases (log(exp(t)) = t is redundant).
    """

    if isinstance(node, (Var, Const)):
        return node

    if isinstance(node, Un):
        child = canonicalize(node.child)
        if child is None:
            return None
        # fold: unary of a pure-constant subtree is a constant
        if not child.has_var():
            return Const()
        if isinstance(child, Un) and (node.op, child.op) in _FORBIDDEN_UNARY_PAIRS:
            return None
        # parameter absorption:  exp(t + c) == c'*exp(t),  log(c*t) == c' + log(t),
        # sqrt(c*t) == c'*sqrt(t), square(c*t) == c'*square(t)  -- all covered by
        # same-size skeletons, so reject.
        if node.op == "exp" and isinstance(child, Bin) and child.op == "add":
            terms = []
            _flatten("add", child, terms)
            if any(not t.has_var() for t in terms):
                return None
        if node.op in ("log", "sqrt", "square") and isinstance(child, Bin) and child.op == "mul":
            terms = []
            _flatten("mul", child, terms)
            if any(not t.has_var() for t in terms):
                return None
        return Un(node.op, child)

    # binary
    left = canonicalize(node.left)
    right = canonicalize(node.right)
    if left is None or right is None:
        return None
    if not left.has_var() and not right.has_var():
        return Const()

    op = node.op
    if op in ("add", "mul"):
        terms: list[Expr] = []
        _flatten(op, left, terms)
        _flatten(op, right, terms)
        # at most one constant term in an associative chain (c1+c2 -> c)
        if sum(1 for t in terms if isinstance(t, Const)) > 1:
            return None
        if op == "add":
            # t + t == 2t: covered by mul(c, t), never larger
            keys = [t.key() for t in terms if t.has_var()]
            if len(keys) != len(set(keys)):
                return None
        if op == "mul":
            # t * t == square(t) with fewer nodes
            keys = [t.key() for t in terms if t.has_var()]
            if len(keys) != len(set(keys)):
                return None
        # canonical order (constants last, then by key)
        terms.sort(key=lambda t: (not t.has_var(), t.key()))
        out = terms[0]
        for t in terms[1:]:
            out = Bin(op, out, t)
        return out

    if op == "sub":
        # t - c == t + c'  and  t - t == 0
        if isinstance(right, Const):
            return None
        if left.key() == right.key():
            return None
        return Bin(op, left, right)

    if op == "div":
        # t / c == c' * t  and  t / t == 1
        if isinstance(right, Const):
            return None
        if left.key() == right.key():
            return None
        return Bin(op, left, right)

    raise ValueError(op)


# ---------------------------------------------------------------------------
# Enumeration and expansion
# ---------------------------------------------------------------------------

def leaves(d: int) -> list[Expr]:
    return [Var(j) for j in range(d)] + [Const()]


def enumerate_skeletons(size: int, d: int, unary: tuple, binary: tuple,
                        _memo=None) -> list[Expr]:
    """All canonical skeletons with exactly siz operator nodes."""
    if _memo is None:
        _memo = {}
    if size in _memo:
        return _memo[size]
    seen = set()
    out = []

    def add(raw):
        c = canonicalize(raw)
        if c is None or not c.has_var() or c.complexity() != size:
            return
        k = c.key()
        if k not in seen:
            seen.add(k)
            out.append(c)

    if size == 0:
        for leaf in leaves(d):
            if leaf.has_var():
                out.append(leaf)
        _memo[size] = out
        return out

    for t in enumerate_skeletons(size - 1, d, unary, binary, _memo):
        for op in unary:
            add(Un(op, t))
    for k1 in range(size):
        k2 = size - 1 - k1
        if k2 < k1:
            break
        for t1 in enumerate_skeletons(k1, d, unary, binary, _memo):
            for t2 in enumerate_skeletons(k2, d, unary, binary, _memo):
                for op in binary:
                    add(Bin(op, t1, t2))
                    if k1 != k2 and op in ("sub", "div"):
                        add(Bin(op, t2, t1))
    _memo[size] = out
    return out


def _replacements(d: int, unary: tuple, binary: tuple) -> list:
    """Depth-1 subtrees (one operator node) used to grow a skeleton."""
    subs = []
    for leaf in leaves(d):
        for op in unary:
            subs.append(Un(op, leaf))
    for l1 in leaves(d):
        for l2 in leaves(d):
            for op in binary:
                subs.append(Bin(op, l1, l2))
    return subs


def expand(expr: Expr, d: int, unary: tuple, binary: tuple) -> list[Expr]:
    """All canonical skeletons obtained from ``expr`` by one growth move.

    Moves: (i) replace a leaf by a depth-1 subtree, (ii) wrap any node with a
    unary operator, (iii) combine any node with a leaf via a binary operator.
    Every move adds exactly one operator node.
    """
    subs = _replacements(d, unary, binary)
    results = {}

    def rebuild(node, path, replacement):
        """Return expr with the node at ``path`` replaced; off-path subtrees are shared."""
        if not path:
            return replacement
        head, *rest = path
        if isinstance(node, Un):
            return Un(node.op, rebuild(node.child, rest, replacement))
        if head == 0:
            return Bin(node.op, rebuild(node.left, rest, replacement), node.right)
        return Bin(node.op, node.left, rebuild(node.right, rest, replacement))

    def visit(node, path):
        # move (ii)/(iii): wrap this node
        wraps = [Un(op, node) for op in unary]
        for leaf in leaves(d):
            for op in binary:
                wraps.append(Bin(op, node, leaf))
                wraps.append(Bin(op, leaf, node))
        for w in wraps:
            cand = canonicalize(rebuild(expr, path, w)) if path else canonicalize(w)
            if cand is not None and cand.has_var() and cand.complexity() == expr.complexity() + 1:
                results[cand.key()] = cand
        if isinstance(node, (Var, Const)):
            # move (i): replace leaf with a depth-1 subtree
            for s in subs:
                cand = canonicalize(rebuild(expr, path, s))
                if cand is not None and cand.has_var() and cand.complexity() == expr.complexity() + 1:
                    results[cand.key()] = cand
            return
        if isinstance(node, Un):
            visit(node.child, path + [0])
        else:
            visit(node.left, path + [0])
            visit(node.right, path + [1])

    visit(expr, [])
    return list(results.values())
