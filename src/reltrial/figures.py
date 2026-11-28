"""Accuracy-vs-cost figure (F): one panel per task, test metric vs. training cost.

Cost = training seconds on the GPU/CPU the run used, plus LightGBM's feature-building time.
Colour = model family, filled = with label history, hollow = without; the grey step line
is the Pareto frontier (nothing is both cheaper and better).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .evaluate import load_runs, mean_ci  # noqa: E402
from .tables import TASK_SHORT, label  # noqa: E402

# Reference categorical palette, first three slots (validated all-pairs, light mode).
FAMILY = {"LightGBM": "#257de0", "GNN": "#f47230", "RelGNN": "#00b894", "Blend": "#191919"}
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"


def family(model: str) -> str:
    if model.startswith("blend_"):
        return "Blend"
    if model.startswith("lgbm"):
        return "LightGBM"
    return "RelGNN" if model.startswith("relgnn") else "GNN"


SHORT = {"lgbm_table": "LightGBM, own row", "lgbm_agg": "LightGBM + aggregates + history",
         "lgbm_agg_nohist": "LightGBM + aggregates"}


def short(model: str) -> str:
    return SHORT.get(model) or label(model).replace(" (GloVe text)", "")


def has_history(model: str) -> bool:
    if model.startswith("blend_"):
        return "hist" in model.split("__")[1] or not model.split("__")[0].endswith("nohist")
    if model.startswith("lgbm"):
        return model == "lgbm_agg"
    return "hist" in model.split("_")


def points(runs: List[dict], task: str, split: str = "test") -> pd.DataFrame:
    rows = []
    df = pd.DataFrame([r for r in runs if r["task"] == task])
    if df.empty:
        return pd.DataFrame()
    metric = "roc_auc" if df.iloc[0]["task_type"] == "binary_classification" else "nmae"
    for model, g in df.groupby("model"):
        vals = [r.get(metric) for r in g[split]]
        mean, half, n = mean_ci(vals)
        cost = [r["train_seconds"] + (r.get("params") or {}).get("feature_seconds", 0.0)
                for _, r in g.iterrows()]
        rows.append({"model": model, "metric": metric, "mean": mean, "ci": half, "seeds": n,
                     "cost": float(np.mean(cost)), "family": family(model),
                     "history": has_history(model)})
    return pd.DataFrame(rows).sort_values("cost")


def frontier(p: pd.DataFrame, higher: bool) -> pd.DataFrame:
    best, keep = None, []
    for _, r in p.sort_values("cost").iterrows():
        if best is None or (r["mean"] > best if higher else r["mean"] < best):
            best = r["mean"]
            keep.append(r)
    return pd.DataFrame(keep)


def pareto(runs_dir: str, out: str, tasks: Optional[List[str]] = None, split: str = "test",
           exclude: tuple = ("_e10",)) -> Dict[str, pd.DataFrame]:
    runs = [r for r in load_runs(runs_dir) if not any(x in r["model"] for x in exclude)]
    tasks = tasks or [t for t in TASK_SHORT if any(r["task"] == t for r in runs)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), squeeze=False)
    result = {}
    for ax, task in zip(axes[0], tasks):
        p = points(runs, task, split)
        if p.empty:
            continue
        result[task] = p
        higher = p.iloc[0]["metric"] == "roc_auc"
        f = frontier(p, higher)
        xs = list(f["cost"]) + [p["cost"].max() * 1.3]
        ys = list(f["mean"]) + [f["mean"].iloc[-1]]
        ax.step(xs, ys, where="post", color="#bdbdbd", lw=2, zorder=1)
        for _, r in p.iterrows():
            c = FAMILY[r["family"]]
            ax.errorbar(r["cost"], r["mean"], yerr=0 if np.isnan(r["ci"]) else r["ci"], fmt="none",
                        ecolor=c, elinewidth=1, alpha=0.6, zorder=2)
            ax.scatter(r["cost"], r["mean"], s=64, marker="D" if r["family"] == "Blend" else "o",
                       facecolor=c if r["history"] else "white", edgecolor=c, linewidth=2, zorder=3)
        # Direct labels on the frontier only, alternating above / below to avoid collisions.
        for i, (_, r) in enumerate(f.iterrows()):
            up = i % 2 == 0
            ax.annotate(short(r["model"]), (r["cost"], r["mean"]),
                        xytext=(7, 9 if up else -13), textcoords="offset points",
                        fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel("Training cost per seed (seconds, log scale)", color=INK2, fontsize=8)
        name = "AUROC (higher is better)" if higher else "NMAE (lower is better)"
        ax.set_ylabel(f"Test {name}", color=INK2, fontsize=8)
        ax.set_title(f"{TASK_SHORT.get(task, task)}: accuracy vs. cost", loc="left", fontsize=10,
                     color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bdbdbd")
        ax.tick_params(colors=INK2, labelsize=7)
    fams = [f for f in FAMILY if any(f in set(p["family"]) for p in result.values())]
    handles = [Line2D([], [], marker="D" if f == "Blend" else "o", ls="", color=FAMILY[f],
                      markerfacecolor=FAMILY[f], markersize=7, label=f) for f in fams]
    handles += [Line2D([], [], marker="o", ls="", color=INK2, markerfacecolor=INK2, markersize=7,
                       label="with label history"),
                Line2D([], [], marker="o", ls="", color=INK2, markerfacecolor="white", markersize=7,
                       markeredgewidth=2, label="without"),
                Line2D([], [], color="#bdbdbd", lw=2, label="Pareto frontier")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    for task, p in result.items():
        p.to_csv(Path(out).with_name(f"pareto_{task}.csv"), index=False)
    return result
