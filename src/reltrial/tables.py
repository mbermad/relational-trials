"""Turn logged runs into the results table (Markdown for the README, LaTeX for the report)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .evaluate import REFERENCE, load_runs, mean_ci

MODEL_LABELS = {
    "lgbm_table": "LightGBM (trial table)",
    "lgbm_agg_nohist": "LightGBM + aggregates, no label history",
    "lgbm_agg": "LightGBM + aggregates",
    "rdl_sage_notext_e10": "GNN, no text (10 epochs)",
    "rdl_sage_notext": "GNN, no text",
    "rdl_sage_notext_hist": "GNN, no text + label history",
    "rdl_sage": "GNN (GloVe text)",
    "rdl_sage_hist": "GNN (GloVe text) + label history",
    "rdl_sage_tuned": "GNN (GloVe text), tuned",
    "rdl_sage_hist_tuned": "GNN (GloVe text) + label history, tuned",
    "relgnn": "RelGNN (GloVe text)",
    "relgnn_hist": "RelGNN (GloVe text) + label history",
    "relgnn_notext": "RelGNN, no text",
    "relgnn_notext_hist": "RelGNN, no text + label history",
    "relgt": "RelGT",
}
MODEL_ORDER = list(MODEL_LABELS)


def label(model: str) -> str:
    """Readable name for any run name, including combinations not listed above."""
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    if model.startswith("blend_") and "__" in model:
        a, b = model[len("blend_"):].split("__", 1)
        return f"Blend: {label(a)} & {label(b)}"
    if model.endswith("_masked"):
        return label(model[:-len("_masked")]) + ", history masked at inference"
    parts = model.split("_")
    base = {"rdl": "GNN", "relgnn": "RelGNN", "lgbm": "LightGBM"}.get(parts[0], parts[0])
    extra = []
    if "notext" in parts:
        extra.append("no text")
    elif base in ("GNN", "RelGNN"):
        extra.append("GloVe text")
    name = f"{base} ({', '.join(extra)})" if extra else base
    if "hist" in parts:
        name += " + label history"
    if "tuned" in parts:
        name += ", tuned"
    for p in parts:
        if p.startswith("e") and p[1:].isdigit():
            name += f" ({p[1:]} epochs)"
        if p.startswith("L") and p[1:].isdigit():
            name += f" ({p[1:]} layers)"
    return name
TASK_ORDER = ["study-outcome", "study-adverse", "driver-dnf", "driver-top3", "driver-position"]
TASK_SHORT = {"study-outcome": "Outcome", "study-adverse": "Adverse", "driver-dnf": "DNF",
              "driver-top3": "Top-3", "driver-position": "Position"}


def columns_for(runs: List[dict]):
    """(task, split, metric, header, higher_is_better) for the tasks present in ``runs``.

    rel-trial keeps the milestone layout: outcome AUROC val/test, adverse NMAE val/test,
    adverse MAE test.
    """
    types = {r["task"]: r["task_type"] for r in runs}
    tasks = [t for t in TASK_ORDER if t in types] + sorted(set(types) - set(TASK_ORDER))
    cols = []
    for t in tasks:
        short = TASK_SHORT.get(t, t)
        if types[t] == "binary_classification":
            cols += [(t, s, "roc_auc", f"{short} AUROC ({s})", True) for s in ("val", "test")]
        else:
            cols += [(t, s, "nmae", f"{short} NMAE ({s})", False) for s in ("val", "test")]
            cols.append((t, "test", "mae", f"{short} MAE (test)", False))
    return cols


def aggregate(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for (task, model), group in pd.DataFrame(runs).groupby(["task", "model"]):
        for split in ("val", "test"):
            metrics = set().union(*[r.keys() for r in group[split]])
            for m in metrics:
                mean, half, n = mean_ci([r.get(m) for r in group[split]])
                rows.append({"task": task, "model": model, "split": split, "metric": m,
                             "mean": mean, "ci95": half, "seeds": n})
        mean, half, n = mean_ci(group["train_seconds"])
        rows.append({"task": task, "model": model, "split": "-", "metric": "train_seconds",
                     "mean": mean, "ci95": half, "seeds": n})
    return pd.DataFrame(rows)


def _fmt(mean: float, half: float, digits: int = 3) -> str:
    if mean is None or math.isnan(mean):
        return "--"
    if half is None or math.isnan(half):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {half:.{digits}f}"


def _cell(agg: pd.DataFrame, model: str, task: str, split: str, metric: str):
    hit = agg[(agg.model == model) & (agg.task == task) & (agg.split == split) & (agg.metric == metric)]
    if hit.empty:
        return None, None, 0
    r = hit.iloc[0]
    return r["mean"], r["ci95"], int(r["seeds"])


def build(runs_dir: str = "results/runs", out_dir: str = "results/tables") -> Dict:
    runs = load_runs(runs_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not runs:
        print("No runs found yet.")
        return {}
    agg = aggregate(runs)
    agg.to_csv(out / "aggregate.csv", index=False)
    COLUMNS = columns_for(runs)
    tasks = list(dict.fromkeys(c[0] for c in COLUMNS))
    models = [m for m in MODEL_ORDER if m in set(agg.model)] + \
             sorted(set(agg.model) - set(MODEL_ORDER))

    # Best value per column (for bolding).
    best = {}
    for task, split, metric, _, higher in COLUMNS:
        vals = [(_cell(agg, m, task, split, metric)[0], m) for m in models]
        vals = [(v, m) for v, m in vals if v is not None and not math.isnan(v)]
        if vals:
            best[(task, split, metric)] = (max if higher else min)(vals)[1]

    # Markdown
    md = ["| Model | " + " | ".join(c[3] for c in COLUMNS) + " | Seeds |",
          "|---|" + "---|" * (len(COLUMNS) + 1)]
    for m in models:
        cells, seeds = [], 0
        for task, split, metric, _, _ in COLUMNS:
            mean, half, n = _cell(agg, m, task, split, metric)
            seeds = max(seeds, n)
            txt = _fmt(mean, half) if mean is not None else "--"
            cells.append(f"**{txt}**" if best.get((task, split, metric)) == m else txt)
        md.append(f"| {label(m)} | " + " | ".join(cells) + f" | {seeds} |")
    ref = []
    for m in ("lgbm_table", "rdl_sage"):
        if not any((c[0], m) in REFERENCE for c in COLUMNS):
            continue
        cells = [f"{REFERENCE[(t, m)]:.3f}" if sp == "test" and met != "mae" and (t, m) in REFERENCE
                 else "--" for t, sp, met, _, _ in COLUMNS]
        ref.append(f"| *RelBench reference: {MODEL_LABELS[m]}* | " + " | ".join(cells) + " | 1 |")
    md += ref
    md.append("")
    md.append("Mean ± 95% CI over seeds. AUROC: higher is better. NMAE = MAE / train std "
              "(RelBench v3's regression metric): lower is better.")
    (out / "main.md").write_text("\n".join(md) + "\n")

    # LaTeX body rows (the report wraps them in a tabular)
    tex = []
    for m in models:
        cells = []
        for task, split, metric, _, _ in COLUMNS:
            mean, half, _ = _cell(agg, m, task, split, metric)
            if mean is None or math.isnan(mean):
                cells.append("--")
                continue
            digits = 2 if metric == "mae" else 3
            body = f"{mean:.{digits}f}" + ("" if half is None or math.isnan(half)
                                          else f"\\,{{\\scriptsize$\\pm${half:.{digits}f}}}")
            cells.append(f"\\textbf{{{body}}}" if best.get((task, split, metric)) == m else body)
        tex.append(label(m).replace("&", "\\&") + " & " + " & ".join(cells) + r" \\")
    tex.append(r"\midrule")
    for m in ("lgbm_table", "rdl_sage"):
        if not any((c[0], m) in REFERENCE for c in COLUMNS):
            continue
        cells = [f"{REFERENCE[(t, m)]:.3f}" if sp == "test" and met != "mae" and (t, m) in REFERENCE
                 else "--" for t, sp, met, _, _ in COLUMNS]
        tex.append(f"\\textit{{RelBench ref.: {MODEL_LABELS[m]}}} & " + " & ".join(cells) + " \\\\")
    (out / "main_rows.tex").write_text("\n".join(tex) + "\n")

    # Runtime table rows
    rt = []
    for m in models:
        cells = []
        for task in tasks:
            mean, _, _ = _cell(agg, m, task, "-", "train_seconds")
            cells.append("--" if mean is None or math.isnan(mean) else f"{mean:.0f}")
        rt.append(label(m) + " & " + " & ".join(cells) + r" \\")
    (out / "runtime_rows.tex").write_text("\n".join(rt) + "\n")

    summary = {"models": models, "n_runs": len(runs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(md))
    return summary
