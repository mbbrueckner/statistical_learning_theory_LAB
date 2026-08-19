# AGML Lab Final Project — Equation Discovery

Symbolic regression for extrapolation: a systematic, complexity-ordered search
over expression skeletons with a polynomial background solved by variable
projection, selected via extrapolation-shaped cross-validation.

## Setup

```bash
cd final_project
uv venv .venv --python 3.12          # or: python3.12 -m venv .venv
uv pip install -r requirements.txt --python .venv/bin/python
```

## Reproduce everything

```bash
# all experiments, then figures + tables
OMP_NUM_THREADS=1 .venv/bin/python src/experiments/run_experiments.py --jobs 8
.venv/bin/python src/experiments/make_figures.py

# report
cd report && latexmk -pdf main.tex
```

All randomness is seeded; re-running reproduces the CSVs in `results/` exactly.
Single experiments can be run on their own via `--exp`:

| `--exp` | what it measures | output |
|---|---|---|
| `hyper` | search hyperparameters selected by cross-validation | `results/hyper.csv` |
| `main` | extrapolation error vs. formula complexity and operator set | `results/main.csv` |
| `noise` | robustness to label noise | `results/noise.csv` |
| `dim2` | two input variables | `results/dim2.csv` |
| `ablation` | model-selection strategies on identical Pareto fronts | `results/ablation.csv` |
| `demo` | the worked example of Figure 1 | `results/demo_*.csv` |

Expect a few hours in total on 4 performance cores; `hyper` and `main` dominate.

## Layout

```
src/sr/                 the method
  expressions.py        expression trees, protected operators, canonicalization
  generator.py          dice-roll formula sampler + extrapolation datasets
  search.py             complexity-ordered search, variable projection,
                        fingerprint deduplication, Pareto front
  selection.py          extrapolation-shaped CV model selection (1-SE rule)
  baselines.py          polynomial ridge / random forest / MLP
  metrics.py            NMSE, functional recovery
src/experiments/        experiment grids, figures, tables
data/                   sampled-formula configs (JSON)
results/                CSVs, figures (figs/), LaTeX tables
report/                 report (main.tex, references.bib)
```
