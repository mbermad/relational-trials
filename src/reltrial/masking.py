"""Deletion test at inference (F1): does a trained model actually use label history?

Train once WITH history, then predict with history columns replaced by their training
mean ("no information"): all of them, and one linked-entity group at a time (sponsor,
facility, ...). A large drop means the model relies on that history; comparing with the
model retrained without history separates "uses it" from "needs it".

Results: ``<results>/masking/<task>/<model>/seed{k}.json`` with one entry per group, and
the all-masked predictions under ``preds/<task>/<model>_masked`` (so the paired bootstrap
picks them up).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from .evaluate import compute_metrics


def history_groups(columns: Sequence[str]) -> Dict[str, List[int]]:
    """Column indices per group: 'all', then one per linked entity ('sponsor', 'own', ...)."""
    cols = list(columns)
    hist = [i for i, c in enumerate(cols) if "_hist_" in c]
    groups: Dict[str, List[int]] = {"all": hist}
    for i in hist:
        groups.setdefault(cols[i].split("_hist_")[0], []).append(i)
    return groups


def evaluate_masked(predict: Callable[[str, List[int]], np.ndarray], groups: Dict[str, List[int]],
                    y: Dict[str, np.ndarray], task_type: str, nmae_std) -> Dict[str, Dict]:
    """``predict(split, cols_to_mask)`` -> predictions; returns metrics per group and split,
    plus the all-masked predictions under '_preds'."""
    out: Dict[str, Dict] = {}
    preds_all = {}
    for g, idx in groups.items():
        out[g] = {"n_columns": len(idx)}
        for s in ("val", "test"):
            p = predict(s, idx)
            out[g][s] = compute_metrics(task_type, y[s], p, nmae_std)
            if g == "all":
                preds_all[s] = p
    out["_preds"] = preds_all
    return out


def save(result: Dict, root: str | Path, task: str, model: str, seed: int,
         preds_root: str | Path | None = None, unmasked: Dict | None = None) -> None:
    preds = result.pop("_preds")
    if unmasked is not None:
        result = {"none": unmasked, **result}
    d = Path(root) / task / model
    d.mkdir(parents=True, exist_ok=True)
    (d / f"seed{seed}.json").write_text(json.dumps(result, indent=2, default=float))
    if preds_root:
        pd_ = Path(preds_root) / task / f"{model}_masked"
        pd_.mkdir(parents=True, exist_ok=True)
        for s, p in preds.items():
            np.save(pd_ / f"seed{seed}_{s}.npy", p)


def summarize(root: str | Path, out_dir: str | Path) -> str:
    """Markdown table: mean metric over seeds for each (task, model, group)."""
    import pandas as pd

    rows = []
    for f in sorted(Path(root).glob("*/*/seed*.json")):
        task, model = f.parent.parent.name, f.parent.name
        for g, r in json.loads(f.read_text()).items():
            for s in ("val", "test"):
                m = "roc_auc" if "roc_auc" in r[s] else "nmae"
                rows.append({"task": task, "model": model, "group": g, "split": s,
                             "metric": m, "value": r[s][m]})
    if not rows:
        return ""
    df = pd.DataFrame(rows).groupby(["task", "model", "group", "split", "metric"])["value"] \
        .mean().unstack("split").reset_index()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "masking.csv", index=False)
    lines = ["| Task | Model | History masked | Metric | Val | Test |", "|---|---|---|---|---|---|"]
    for (task, model), g in df.groupby(["task", "model"], sort=False):
        base = g[g.group == "none"]
        order = ["none", "all"] + sorted(set(g.group) - {"none", "all"})
        for grp in order:
            r = g[g.group == grp]
            if r.empty:
                continue
            r = r.iloc[0]

            def cell(s):
                v = r[s]
                if grp == "none" or base.empty:
                    return f"{v:.3f}"
                return f"{v:.3f} ({v - base.iloc[0][s]:+.3f})"
            lines.append(f"| {task} | {model} | {grp} | {r.metric} | {cell('val')} | {cell('test')} |")
    lines += ["", "History columns replaced by their training mean at inference; the model is "
              "not retrained. In brackets: change vs. no masking (mean over seeds)."]
    md = "\n".join(lines) + "\n"
    (out / "masking.md").write_text(md)
    return md
