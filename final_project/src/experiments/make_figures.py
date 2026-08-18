"""Generate report figures and LaTeX tables from results/*.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS

FIGS = RESULTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

METHODS = {
    "SR (ours)": ("SR (ours)", "#0072B2", "-", "o"),
    "polynomial": ("Polynomial", "#E69F00", "--", "s"),
    "random_forest": ("Random forest", "#009E73", "-.", "^"),
    "mlp": ("MLP", "#CC79A7", ":", "D"),
}
OPSET_LABEL = {"poly": r"$\mathcal{O}_{\mathrm{poly}}$",
               "trig": r"$\mathcal{O}_{\mathrm{trig}}$",
               "full": r"$\mathcal{O}_{\mathrm{full}}$"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def _bin_eff(df):
    bins = [-1, 2, 4, 6, 100]
    labels = ["≤2", "3–4", "5–6", "7+"]
    df = df.copy()
    df["eff_bin"] = pd.cut(df["eff_ops"], bins=bins, labels=labels)
    return df, labels


def fig_demo():
    curves = pd.read_csv(RESULTS / "demo_curves.csv")
    train = pd.read_csv(RESULTS / "demo_train.csv")
    fig, ax = plt.subplots(figsize=(5.6, 2.0))
    ax.axvspan(-5, 5, color="0.92", zorder=0)
    ax.text(0, -3.4, "training domain", ha="center", color="0.45", fontsize=7.5)
    ax.plot(curves["x"], curves["truth"], color="black", lw=1.2, ls=(0, (4, 2)),
            label=r"truth $0.5x\sin x + 0.1x^2$", zorder=6)
    for key, (name, color, ls, _) in METHODS.items():
        ax.plot(curves["x"], curves[key], color=color,
                lw=2.2 if key == "SR (ours)" else 1.6, ls=ls,
                label=name, zorder=4 if key == "SR (ours)" else 2)
    ax.scatter(train["x"], train["y"], s=6, color="0.25", zorder=5,
               label="train points", linewidths=0)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-4.2, 19)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(ncol=3, frameon=False, loc="upper center", handlelength=1.8,
              columnspacing=0.9, fontsize=7)
    fig.savefig(FIGS / "demo.pdf")
    plt.close(fig)


def _median_band(ax, sub, xcol, order=None):
    for key, (name, color, ls, marker) in METHODS.items():
        g = sub[sub.method == key].groupby(xcol, observed=True)["nmse_test"]
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        if order is not None:
            med, lo, hi = med.reindex(order), lo.reindex(order), hi.reindex(order)
        x = np.arange(len(med)) if order is not None else med.index.to_numpy()
        ax.plot(x, med.to_numpy(), color=color, ls=ls, marker=marker, ms=3.5,
                lw=1.6, label=name)
        ax.fill_between(x, lo.to_numpy(), hi.to_numpy(), color=color, alpha=0.13, lw=0)
        if order is not None:
            ax.set_xticks(x, order)
    ax.set_yscale("log")


def fig_main():
    df = pd.read_csv(RESULTS / "main.csv")
    df, labels = _bin_eff(df)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.0))

    df["success"] = df["nmse_test"] < 1e-3
    for key, (name, color, ls, marker) in METHODS.items():
        g = df[df.method == key].groupby("eff_bin", observed=True)["success"].mean()
        g = g.reindex(labels)
        axes[0].plot(np.arange(len(g)), 100 * g.to_numpy(), color=color, ls=ls,
                     marker=marker, ms=3.5, lw=1.6, label=name)
    axes[0].set_xticks(np.arange(len(labels)), labels)
    axes[0].set_xlabel("effective complexity of ground truth")
    axes[0].set_ylabel(r"success rate [%]")
    axes[0].set_ylim(-3, 105)
    axes[0].legend(frameon=False, handlelength=1.8)
    axes[0].set_title(r"(a) extrapolation success (NMSE $< 10^{-3}$)", fontsize=8.5)

    sr = df[df.method == "SR (ours)"]
    for op_set, color, marker in [("poly", "#56B4E9", "o"), ("trig", "#D55E00", "s"),
                                  ("full", "#000000", "^")]:
        g = sr[sr.op_set == op_set].groupby("eff_bin", observed=True)["recovered"].mean()
        g = g.reindex(labels)
        axes[1].plot(np.arange(len(g)), 100 * g.to_numpy(), color=color, marker=marker,
                     ms=3.5, lw=1.6, label=OPSET_LABEL[op_set])
    axes[1].set_xticks(np.arange(len(labels)), labels)
    axes[1].set_xlabel("effective complexity of ground truth")
    axes[1].set_ylabel("recovery rate [%]")
    axes[1].set_ylim(0, 105)
    axes[1].legend(frameon=False, handlelength=1.8)
    axes[1].set_title("(b) SR formula recovery", fontsize=8.5)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGS / "main.pdf")
    plt.close(fig)


def fig_noise():
    df = pd.read_csv(RESULTS / "noise.csv")
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.0))
    _median_band(axes[0], df, "noise")
    axes[0].set_ylim(1e-7, 80)
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].set_xticks([0, 0.01, 0.05, 0.1, 0.2], ["0", "0.01", "0.05", "0.1", "0.2"])
    axes[0].set_xlabel(r"relative noise level $\sigma_{\mathrm{rel}}$")
    axes[0].set_ylabel(r"NMSE$_{\mathrm{test}}$ (median, IQR)")
    axes[0].legend(frameon=False, handlelength=1.8)
    axes[0].set_title("(a) extrapolation error vs. noise", fontsize=8.5)

    sr = df[df.method == "SR (ours)"]
    for thr, color, marker, label in [(1e-3, "#0072B2", "o", r"NMSE $< 10^{-3}$"),
                                      (0.1, "#56B4E9", "s", r"NMSE $< 0.1$")]:
        g = sr.assign(s=sr.nmse_test < thr).groupby("noise")["s"].mean() * 100
        axes[1].plot(g.index, g.to_numpy(), color=color, marker=marker, ms=3.5,
                     lw=1.6, label=label)
    axes[1].set_xscale("symlog", linthresh=0.01)
    axes[1].set_xticks([0, 0.01, 0.05, 0.1, 0.2], ["0", "0.01", "0.05", "0.1", "0.2"])
    axes[1].set_xlabel(r"relative noise level $\sigma_{\mathrm{rel}}$")
    axes[1].set_ylabel("SR success rate [%]")
    axes[1].set_ylim(0, 105)
    axes[1].legend(frameon=False, handlelength=1.8)
    axes[1].set_title("(b) SR success vs. noise", fontsize=8.5)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGS / "noise.pdf")
    plt.close(fig)


def fig_front():
    front = pd.read_csv(RESULTS / "demo_front.csv")
    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    ax.plot(front["core_k"], front["train_mse"], color="#0072B2",
            marker="o", ms=4, lw=1.6)
    ax.set_yscale("log")
    ax.set_xticks(front["core_k"])
    ax.set_xlabel("core complexity $k$")
    ax.set_ylabel("training MSE")
    sel = front[front["selected"]]
    if len(sel):
        ax.scatter(sel["core_k"], sel["train_mse"], s=70, facecolors="none",
                   edgecolors="#D55E00", linewidths=1.4, zorder=5)
        ax.annotate("selected", (sel["core_k"].iloc[0], sel["train_mse"].iloc[0]),
                    textcoords="offset points", xytext=(6, 8), fontsize=7.5,
                    color="#D55E00")
    fig.savefig(FIGS / "front.pdf")
    plt.close(fig)


def _fmt(x):
    if not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    if -2 <= exp <= 2:
        return f"{x:.2f}" if abs(x) >= 0.1 else f"{x:.3f}"
    return f"$10^{{{exp}}}$" if abs(x / 10 ** exp - 1) < 0.05 else \
        f"${x / 10 ** exp:.1f}\\!\\cdot\\!10^{{{exp}}}$"


def tab_main():
    rows = []
    main = pd.read_csv(RESULTS / "main.csv")
    dim2 = pd.read_csv(RESULTS / "dim2.csv")
    for (label, df) in [("d=1", main), ("d=2", dim2)]:
        for op_set in ("poly", "trig", "full"):
            sub = df[df.op_set == op_set]
            if not len(sub):
                continue
            row = {"setting": f"{label}, {op_set}"}
            for key, (name, *_ ) in METHODS.items():
                s = sub[sub.method == key]["nmse_test"]
                row[name] = f"{_fmt(s.median())}"
                if key == "SR (ours)":
                    row["Recovery"] = f"{100 * sub[sub.method == key]['recovered'].mean():.0f}\\%"
            rows.append(row)
    t = pd.DataFrame(rows)
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Setting & SR (ours) & Polynomial & Random forest & MLP & Recovery \\",
             r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r['setting']} & {r['SR (ours)']} & {r['Polynomial']} & "
                     f"{r['Random forest']} & {r['MLP']} & {r['Recovery']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS / "tab_main.tex").write_text("\n".join(lines))


def tab_ablation():
    df = pd.read_csv(RESULTS / "ablation.csv")
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"$\sigma_{\mathrm{rel}}$ & Selection & med.\ NMSE & p90 NMSE & "
             r"$\Pr[\nmse > 1]$ & med.\ $C$ \\",
             r"\midrule"]
    for noise in sorted(df.noise.unique()):
        for sel in ("extrap-CV", "interior-CV", "train-MSE"):
            sub = df[(df.noise == noise) & (df.selection == sel)]
            lines.append(f"{noise:g} & {sel} & {_fmt(sub.nmse_test.median())} & "
                         f"{_fmt(sub.nmse_test.quantile(0.9))} & "
                         f"{(sub.nmse_test > 1).mean():.2f} & "
                         f"{sub.complexity.median():.0f} \\\\")
        lines.append(r"\midrule" if noise != sorted(df.noise.unique())[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (RESULTS / "tab_ablation.tex").write_text("\n".join(lines))


CONFIG_ORDER = ["default", "k_ex=2", "k_ex=4", "dedup=affine", "dedup=background",
                "k_ex=4+dedup", "beam=5", "beam=40", "poly_deg=0", "poly_deg=1",
                "poly_deg=3"]


def _cv_selected(df: pd.DataFrame) -> pd.DataFrame:
    """Per dataset, the configuration with the best CV score; ties (frequent,
    since many configurations find the same formula) go to the cheaper one.
    """
    d = df.sort_values(["cv_score", "n_fitted"])
    return d.groupby("seed", as_index=False).first()


def tab_hyper():
    df = pd.read_csv(RESULTS / "hyper.csv")
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Configuration & med.\ NMSE$_{\mathrm{test}}$ & Recovery & med.\ \#fits & "
             r"med.\ time [s] & total time [s] \\",
             r"\midrule"]
    for name in CONFIG_ORDER:
        s = df[df.config == name]
        if not len(s):
            continue
        label = name.replace("_", r"\_")
        if name == "default":
            label = r"default ($k_{\mathrm{ex}}{=}3$, no dedup, $B{=}15$, deg $2$)"
        lines.append(f"{label} & {_fmt(s.nmse_test.median())} & "
                     f"{100 * s.recovered.mean():.0f}\\% & {s.n_fitted.median():.0f} & "
                     f"{s.time_s.median():.1f} & {s.time_s.sum():.0f} \\\\")
    sel = _cv_selected(df)
    oracle = df.sort_values("nmse_test").groupby("seed", as_index=False).first()
    lines += [r"\midrule",
              f"CV-selected per dataset & {_fmt(sel.nmse_test.median())} & "
              f"{100 * sel.recovered.mean():.0f}\\% & {sel.n_fitted.median():.0f} & "
              f"{sel.time_s.median():.1f} & {sel.time_s.sum():.0f} \\\\",
              f"oracle (best test error) & {_fmt(oracle.nmse_test.median())} & "
              f"{100 * oracle.recovered.mean():.0f}\\% & -- & -- & -- \\\\",
              r"\bottomrule", r"\end{tabular}"]
    (RESULTS / "tab_hyper.tex").write_text("\n".join(lines))


def hyper_summary():
    df = pd.read_csv(RESULTS / "hyper.csv")
    print(f"\n== hyperparameter study: {df.seed.nunique()} datasets x "
          f"{df.config.nunique()} configs ==")
    agg = df.groupby("config").agg(
        nmse=("nmse_test", "median"), recovery=("recovered", "mean"),
        fits=("n_fitted", "median"), t=("time_s", "median"),
        total_t=("time_s", "sum")).reindex(CONFIG_ORDER)
    print(agg.round(4))
    sel = _cv_selected(df)
    oracle = df.sort_values("nmse_test").groupby("seed", as_index=False).first()
    print(f"\nCV-selected : recovery={sel.recovered.mean():.3f} "
          f"median nmse={sel.nmse_test.median():.3g} median fits={sel.n_fitted.median():.0f}")
    print(f"oracle      : recovery={oracle.recovered.mean():.3f} "
          f"median nmse={oracle.nmse_test.median():.3g}")
    best = agg.recovery.idxmax()
    print(f"best fixed config by recovery: {best} ({agg.recovery.max():.3f})")
    print("\nconfig chosen by CV, counts:", sel.config.value_counts().to_dict())


def summary():
    """Print headline numbers for the report text."""
    main = pd.read_csv(RESULTS / "main.csv")
    print("== main (d=1) ==")
    print(main.groupby("method")["nmse_test"].median())
    sr = main[main.method == "SR (ours)"]
    print("SR recovery overall:", sr.recovered.mean())
    print("SR success (nmse<0.1):", (sr.nmse_test < 0.1).mean(),
          " poly-baseline:", (main[main.method == "polynomial"].nmse_test < 0.1).mean())
    print("median SR time:", sr.time_s.median(), "max:", sr.time_s.max())
    print("chosen configuration:", sr["config"].value_counts().to_dict())
    for f in ("noise", "dim2", "ablation"):
        try:
            df = pd.read_csv(RESULTS / f"{f}.csv")
            print(f"== {f}: {len(df)} rows ==")
        except FileNotFoundError:
            print(f"== {f}: missing ==")


if __name__ == "__main__":
    fig_demo()
    fig_main()
    fig_noise()
    fig_front()
    tab_main()
    tab_ablation()
    if (RESULTS / "hyper.csv").exists():
        tab_hyper()
        hyper_summary()
    summary()
    print("figures written to", FIGS)
