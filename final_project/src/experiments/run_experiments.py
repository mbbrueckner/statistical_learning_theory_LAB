"""Run all experiments.

Usage:  python src/experiments/run_experiments.py --exp main noise dim2 ablation demo
        (default: all; results land in results/*.csv, figures via make_figures.py)
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from common import (
    DATA, RESULTS, evaluate_dataset, run_grid, run_sr, save_configs,
)

BASE_SEED = 20260712


# ---------------------------------------------------------------------------
# Experiment grids
# ---------------------------------------------------------------------------

def configs_main(n_formulas: int):
    cfgs = []
    i = 0
    for op_set in ("poly", "trig", "full"):
        for n_ops in (2, 4, 6, 8):
            for m in range(n_formulas):
                cfgs.append({"seed": BASE_SEED + i, "n_ops": n_ops, "d": 1,
                             "op_set": op_set, "noise": 0.0})
                i += 1
    return cfgs


def configs_noise(n_formulas: int):
    cfgs = []
    i = 0
    for noise in (0.0, 0.01, 0.05, 0.1, 0.2):
        for m in range(n_formulas):
            cfgs.append({"seed": BASE_SEED + 10_000 + i, "n_ops": 4, "d": 1,
                         "op_set": "full", "noise": noise})
            i += 1
    return cfgs


def configs_dim2(n_formulas: int):
    cfgs = []
    i = 0
    for op_set in ("trig", "full"):
        for n_ops in (2, 4, 6):
            for m in range(n_formulas):
                cfgs.append({"seed": BASE_SEED + 20_000 + i, "n_ops": n_ops, "d": 2,
                             "op_set": op_set, "noise": 0.0})
                i += 1
    return cfgs


def configs_ablation(n_formulas: int):
    cfgs = []
    i = 0
    for noise in (0.0, 0.1):
        for n_ops in (4, 6):
            for m in range(n_formulas):
                cfgs.append({"seed": BASE_SEED + 30_000 + i, "n_ops": n_ops, "d": 1,
                             "op_set": "full", "noise": noise})
                i += 1
    return cfgs


# ---------------------------------------------------------------------------
# Hyperparameter study: one-factor-at-a-time grid around the default
# ---------------------------------------------------------------------------

HYPER_DEFAULT = dict(k_exhaustive=3, dedup="none", beam_width=15, poly_degree=2)
HYPER_VARIANTS = [
    ("default", {}),
    ("k_ex=2", {"k_exhaustive": 2}),
    ("k_ex=4", {"k_exhaustive": 4}),
    ("dedup=affine", {"dedup": "affine"}),
    ("dedup=background", {"dedup": "background"}),
    ("k_ex=4+dedup", {"k_exhaustive": 4, "dedup": "background"}),
    ("beam=5", {"beam_width": 5}),
    ("beam=40", {"beam_width": 40}),
    ("poly_deg=0", {"poly_degree": 0}),
    ("poly_deg=1", {"poly_degree": 1}),
    ("poly_deg=3", {"poly_degree": 3}),
]


def configs_hyper(n_formulas: int):
    cfgs = []
    i = 0
    for op_set in ("trig", "full"):
        for n_ops in (4, 6, 8):
            for m in range(n_formulas):
                cfgs.append({"seed": BASE_SEED + 40_000 + i, "n_ops": n_ops, "d": 1,
                             "op_set": op_set, "noise": 0.0})
                i += 1
    return cfgs


def evaluate_hyper(cfg: dict) -> list[dict]:
    """Run every hyperparameter configuration on one dataset.

    Records the test error *and* the extrapolation-shaped CV score of the
    selected model, so that configurations can afterwards be chosen per
    dataset by cross-validation alone (never touching test data).
    """
    warnings.filterwarnings("ignore")
    from sr import sample_dataset
    from sr.expressions import to_string
    from sr.metrics import mse, nmse, functional_recovery

    ds = sample_dataset(seed=cfg["seed"], n_ops=cfg["n_ops"], d=cfg["d"],
                        op_set=cfg["op_set"], noise=cfg["noise"])
    base = {"seed": cfg["seed"], "d": ds.d, "op_set": ds.op_set, "n_ops": ds.n_ops,
            "eff_ops": ds.meta["effective_ops"], "truth": to_string(ds.expr, ds.theta)}
    rows = []
    for name, override in HYPER_VARIANTS:
        params = dict(HYPER_DEFAULT)
        params.update(override)
        stats: dict = {}
        t0 = time.time()
        chosen, info, _ = run_sr(ds, seed=0, params={**params, "stats": stats},
                                 portfolio=None)   # one configuration per row
        elapsed = time.time() - t0
        pred = chosen.predict(ds.X_test)
        rows.append({**base, "config": name, **params,
                     "mse_test": mse(pred, ds.y_test),
                     "nmse_test": nmse(pred, ds.y_test),
                     "cv_score": info.get("cv_score", np.nan),
                     "recovered": functional_recovery(chosen.predict, ds),
                     "complexity": chosen.complexity, "n_fitted": stats.get("n_fitted"),
                     "n_skipped": stats.get("n_skipped"), "time_s": elapsed})
    return rows


# ---------------------------------------------------------------------------
# Ablation worker: same search, four model-selection strategies
# ---------------------------------------------------------------------------

def evaluate_ablation(cfg: dict) -> list[dict]:
    warnings.filterwarnings("ignore")
    import sys
    from common import SRC  # noqa
    from sr import sample_dataset
    from sr.expressions import to_string
    from sr.metrics import mse, nmse, functional_recovery
    from sr.selection import extrapolation_folds, interior_folds, select

    ds = sample_dataset(seed=cfg["seed"], n_ops=cfg["n_ops"], d=cfg["d"],
                        op_set=cfg["op_set"], noise=cfg["noise"])
    base = {"seed": cfg["seed"], "d": ds.d, "op_set": ds.op_set, "n_ops": ds.n_ops,
            "eff_ops": ds.meta["effective_ops"], "noise": ds.noise,
            "truth": to_string(ds.expr, ds.theta)}

    # A single fixed configuration, so that all three selection strategies
    # arbitrate over literally the same Pareto front.
    chosen, info, t_sr = run_sr(ds, seed=0, portfolio=None,
                                params={"k_exhaustive": 4, "dedup": "background"})
    front = info["front"]

    variants = {"extrap-CV": (chosen, info)}
    interior = select(front, ds.X_train, ds.y_train,
                      folds=interior_folds(ds.X_train, seed=0), seed=0)
    variants["interior-CV"] = interior
    train_best = min(front, key=lambda c: c.mse)
    variants["train-MSE"] = (train_best, {})

    rows = []
    for name, (cand, inf) in variants.items():
        pred = cand.predict(ds.X_test)
        rows.append({**base, "selection": name,
                     "mse_test": mse(pred, ds.y_test),
                     "nmse_test": nmse(pred, ds.y_test),
                     "recovered": functional_recovery(cand.predict, ds),
                     "model": cand.to_string(), "complexity": cand.complexity,
                     "cv_score": inf.get("cv_score"), "time_s": t_sr})
    return rows


# ---------------------------------------------------------------------------
# Demo: the slide example, one dataset, all methods, dense predictions
# ---------------------------------------------------------------------------

def run_demo():
    warnings.filterwarnings("ignore")
    import pandas as pd
    from sr.baselines import BASELINES
    from sr.generator import Dataset
    from sr.expressions import OP_SETS
    from sr import pareto_search, select
    from sr.selection import extrapolation_folds
    from sr.metrics import mse

    rng = np.random.default_rng(42)
    a, b = 5.0, 8.0
    X_train = rng.uniform(-a, a, (128, 1))

    def f(x):
        return 0.5 * x * np.sin(x) + 0.1 * x ** 2

    y_train = f(X_train[:, 0])
    grid = np.linspace(-b, b, 800).reshape(-1, 1)

    ops = OP_SETS["full"]
    front = pareto_search(X_train, y_train, ops["unary"], ops["binary"],
                          k_exhaustive=4, k_max=7, dedup="background", seed=0)
    chosen, info = select(front, X_train, y_train)
    out = {"x": grid[:, 0], "truth": f(grid[:, 0]), "SR (ours)": chosen.predict(grid)}
    print("demo SR model:", chosen.to_string())

    folds = extrapolation_folds(X_train)
    for name, fit in BASELINES.items():
        model, _ = fit(X_train, y_train, folds=folds)
        out[name] = model.predict(grid)

    df = pd.DataFrame(out)
    df.to_csv(RESULTS / "demo_curves.csv", index=False)
    pd.DataFrame({"x": X_train[:, 0], "y": y_train}).to_csv(
        RESULTS / "demo_train.csv", index=False)

    # Pareto front of a *noisy* variant for the analysis figure (the noise-free
    # search stops at the exact fit, which makes for a degenerate front)
    y_noisy = y_train + 0.05 * np.std(y_train) * rng.standard_normal(len(y_train))
    front_n = pareto_search(X_train, y_noisy, ops["unary"], ops["binary"],
                            k_exhaustive=4, k_max=7, dedup="background",
                            mse_tol=0.0, seed=0)
    chosen_n, info_n = select(front_n, X_train, y_noisy)
    rows = [{"core_k": c.core_complexity, "total_k": c.complexity,
             "train_mse": c.mse, "model": c.to_string(),
             "selected": c is chosen_n} for c in front_n]
    pd.DataFrame(rows).to_csv(RESULTS / "demo_front.csv", index=False)
    print("noisy demo chosen:", chosen_n.to_string())
    print("wrote demo CSVs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+",
                    default=["hyper", "main", "noise", "dim2", "ablation", "demo"])
    ap.add_argument("--n-formulas", type=int, default=25)
    ap.add_argument("--n-formulas-2d", type=int, default=15)
    ap.add_argument("--n-formulas-hyper", type=int, default=8)
    ap.add_argument("--n-formulas-ablation", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    t0 = time.time()
    if "hyper" in args.exp:
        cfgs = configs_hyper(args.n_formulas_hyper)
        save_configs(cfgs, "hyper")
        run_grid(cfgs, evaluate_hyper, RESULTS / "hyper.csv", n_jobs=args.jobs)
    if "main" in args.exp:
        cfgs = configs_main(args.n_formulas)
        save_configs(cfgs, "main")
        run_grid(cfgs, evaluate_dataset, RESULTS / "main.csv", n_jobs=args.jobs)
    if "noise" in args.exp:
        cfgs = configs_noise(args.n_formulas)
        save_configs(cfgs, "noise")
        run_grid(cfgs, evaluate_dataset, RESULTS / "noise.csv", n_jobs=args.jobs)
    if "dim2" in args.exp:
        cfgs = configs_dim2(args.n_formulas_2d)
        save_configs(cfgs, "dim2")
        run_grid(cfgs, evaluate_dataset, RESULTS / "dim2.csv", n_jobs=args.jobs)
    if "ablation" in args.exp:
        cfgs = configs_ablation(args.n_formulas_ablation)
        save_configs(cfgs, "ablation")
        run_grid(cfgs, evaluate_ablation, RESULTS / "ablation.csv", n_jobs=args.jobs)
    if "demo" in args.exp:
        run_demo()
    print(f"total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
