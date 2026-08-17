"""Shared experiment infrastructure."""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sr import pareto_search, select, sample_dataset  # noqa: E402
from sr.expressions import OP_SETS, to_string  # noqa: E402
from sr.baselines import BASELINES  # noqa: E402
from sr.metrics import mse, nmse, functional_recovery  # noqa: E402
from sr.selection import extrapolation_folds, interior_folds  # noqa: E402

ROOT = SRC.parent            # final_project/
RESULTS = ROOT / "results"
DATA = ROOT / "data"

SEARCH_PARAMS_1D = dict(k_exhaustive=3, k_max=7, beam_width=15, n_random_parents=5,
                        n_restarts=3, poly_degree=2, dedup="none")
SEARCH_PARAMS_2D = dict(k_exhaustive=2, k_max=6, beam_width=15, n_random_parents=5,
                        n_restarts=3, poly_degree=2, dedup="none")

# Chosen by the hyperparameter study in results/hyper.csv: this pair reaches the
# oracle recovery rate of the full 11-configuration grid at a ninth of its cost.
PORTFOLIO = [
    ("k_ex=4+dedup", {"k_exhaustive": 4, "dedup": "background"}),
    ("poly_deg=3", {"poly_degree": 3}),
]


def search_params(d: int) -> dict:
    return dict(SEARCH_PARAMS_1D if d == 1 else SEARCH_PARAMS_2D)


def run_sr_single(ds, seed: int = 0, folds=None, params=None):
    """Run search + selection once with a fixed configuration."""
    p = search_params(ds.d)
    if params:
        p.update(params)
    ops = OP_SETS[ds.op_set]
    t0 = time.time()
    front = pareto_search(ds.X_train, ds.y_train, ops["unary"], ops["binary"],
                          seed=seed, **p)
    chosen, info = select(front, ds.X_train, ds.y_train, folds=folds, seed=seed)
    info["front"] = front
    return chosen, info, time.time() - t0


def run_sr(ds, seed: int = 0, folds=None, params=None, portfolio=PORTFOLIO):
    """Full pipeline: run every portfolio configuration and keep the model with
    the best cross-validated extrapolation score (ties broken towards the
    cheaper configuration).  Test data is never consulted."""
    if not portfolio:
        return run_sr_single(ds, seed=seed, folds=folds, params=params)
    best = None
    total = 0.0
    for name, override in portfolio:
        p = dict(override)
        if params:
            p.update(params)
        stats: dict = {}
        cand, info, t = run_sr_single(ds, seed=seed, folds=folds,
                                      params={**p, "stats": stats})
        total += t
        key = (info.get("cv_score", np.inf), stats.get("n_fitted", 0))
        if best is None or key < best[0]:
            info["config"] = name
            info["stats"] = stats
            best = (key, cand, info)
    return best[1], best[2], total


def evaluate_dataset(cfg: dict) -> list[dict]:
    """Sample one dataset, run SR + baselines, return one result row per method."""
    warnings.filterwarnings("ignore")
    ds = sample_dataset(seed=cfg["seed"], n_ops=cfg["n_ops"], d=cfg["d"],
                        op_set=cfg["op_set"], noise=cfg["noise"])
    base = {
        "seed": cfg["seed"], "d": ds.d, "op_set": ds.op_set, "n_ops": ds.n_ops,
        "eff_ops": ds.meta["effective_ops"], "noise": ds.noise,
        "truth": to_string(ds.expr, ds.theta),
        "y_test_var": float(np.var(ds.y_test)),
    }
    rows = []

    chosen, info, t_sr = run_sr(ds, seed=0)
    pred = chosen.predict(ds.X_test)
    rows.append({**base, "method": "SR (ours)",
                 "mse_test": mse(pred, ds.y_test), "nmse_test": nmse(pred, ds.y_test),
                 "recovered": functional_recovery(chosen.predict, ds),
                 "model": chosen.to_string(), "complexity": chosen.complexity,
                 "config": info.get("config"), "time_s": t_sr})

    folds = extrapolation_folds(ds.X_train)
    for name, fit in BASELINES.items():
        t0 = time.time()
        model, params = fit(ds.X_train, ds.y_train, folds=folds)
        t_b = time.time() - t0
        pred = model.predict(ds.X_test)
        rows.append({**base, "method": name,
                     "mse_test": mse(pred, ds.y_test), "nmse_test": nmse(pred, ds.y_test),
                     "recovered": False, "model": str(params), "complexity": None,
                     "config": None, "time_s": t_b})
    return rows


def run_grid(configs: list[dict], worker, out_csv: Path, n_jobs: int = 8):
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(worker)(c) for c in configs)
    rows = [r for rr in results for r in rr]
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} rows)")
    return df


def save_configs(configs: list[dict], name: str):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / f"{name}_configs.json", "w") as f:
        json.dump(configs, f, indent=1)
