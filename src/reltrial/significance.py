"""Paired bootstrap tests for the comparisons the write-up makes.

For a pair (A, B) on one split, each bootstrap resample of the split's rows is scored for
every seed of both models; the statistic is the difference of the seed-averaged metrics,
oriented so that a positive value means A is better. This accounts for the finite test set
(the main source of noise here) and for seed variation at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

from .data import TaskBundle
from .tables import label


def load_seed_preds(preds_root: str | Path, task: str, model: str, split: str) -> Optional[np.ndarray]:
    files = sorted(Path(preds_root, task, model).glob(f"seed*_{split}.npy"))
    return np.stack([np.load(f) for f in files]) if files else None


def _score(metric: str, y: np.ndarray, p: np.ndarray, std: float) -> float:
    if metric == "roc_auc":
        return roc_auc_score(y, p)
    return mean_absolute_error(y, p) / std


def compare(y: np.ndarray, A: np.ndarray, B: np.ndarray, metric: str, std: float = 1.0,
            n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    sign = 1.0 if metric == "roc_auc" else -1.0  # positive = A better
    rng = np.random.default_rng(seed)
    point = sign * (np.mean([_score(metric, y, a, std) for a in A])
                    - np.mean([_score(metric, y, b, std) for b in B]))
    diffs = []
    n = len(y)
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, n)
        if metric == "roc_auc" and len(np.unique(y[idx])) < 2:
            continue
        a = np.mean([_score(metric, y[idx], s[idx], std) for s in A])
        b = np.mean([_score(metric, y[idx], s[idx], std) for s in B])
        diffs.append(sign * (a - b))
    d = np.asarray(diffs)
    return {"diff": float(point), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)), "p_a_better": float((d > 0).mean())}


def default_pairs(models: Sequence[str]) -> List[Tuple[str, str, str]]:
    """(A, B, question) pairs present in ``models``."""
    have = set(models)
    pairs = []
    for m in sorted(have):
        parts = m.split("_")
        if "hist" in parts:
            base = "_".join(p for p in parts if p != "hist")
            if base in have:
                pairs.append((m, base, "label history"))
    if {"lgbm_agg", "lgbm_agg_nohist"} <= have:
        pairs.append(("lgbm_agg", "lgbm_agg_nohist", "label history"))
    extra = [("rdl_sage", "rdl_sage_notext", "text"),
             ("rdl_sage_tuned", "rdl_sage", "tuning"),
             ("relgnn", "rdl_sage", "RelGNN vs GraphSAGE"),
             ("relgnn_hist", "rdl_sage_hist", "RelGNN vs GraphSAGE (with history)"),
             ("lgbm_agg", "rdl_sage_hist", "LightGBM vs best GNN"),
             ("lgbm_table", "rdl_sage", "LightGBM (own row) vs GNN")]
    pairs += [p for p in extra if p[0] in have and p[1] in have]
    for m in sorted(have):
        if "_ghist" in m:
            base = m.replace("_ghist", "")
            if base in have:
                pairs.append((m, base, "history in the graph"))
            if base + "_hist" in have:
                pairs.append((m, base + "_hist", "history in the graph vs at the head"))
        if m.endswith("_hist_atstart"):
            base = m[: -len("_hist_atstart")]
            for other, q in ((base + "_hist", "history at seed time vs at trial start"),
                             (base + "_ghist", "history at the head vs in the graph, same date")):
                if other in have:
                    pairs.append((other, m, q) if other.endswith("_hist") else (m, other, q))
        if m + "_masked" in have:
            pairs.append((m, m + "_masked", "history used at inference"))
        if m.startswith("blend_") and "__" in m:
            a, b = m[len("blend_"):].split("__", 1)
            pairs += [(m, x, "blend vs single model") for x in (a, b) if x in have]
    return pairs


def run(bundles: Dict[str, TaskBundle], preds_root: str, out_dir: str, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for task, b in bundles.items():
        models = sorted(p.name for p in Path(preds_root, task).glob("*") if p.is_dir())
        metric = "roc_auc" if b.is_classification else "nmae"
        std = 1.0 if b.is_classification else b.nmae_std()
        for a, bb, question in default_pairs(models):
            for split in ("val", "test"):
                y = b.split(split).df[b.task.target_col].to_numpy(dtype=float)
                A = load_seed_preds(preds_root, task, a, split)
                B = load_seed_preds(preds_root, task, bb, split)
                if A is None or B is None or A.shape[1] != len(y) or B.shape[1] != len(y):
                    continue
                r = compare(y, A, B, metric, std, n_boot=n_boot)
                rows.append({"task": task, "question": question, "A": a, "B": bb, "split": split,
                             "metric": metric, "seeds": f"{len(A)}v{len(B)}", **r})
                print(f"  {task} {split}: {label(a)} vs {label(bb)}: {r['diff']:+.4f} "
                      f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  P(A better)={r['p_a_better']:.3f}")
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "bootstrap.csv", index=False)
    md = ["| Task | Question | A | B | Split | A − B (better = +) | 95% CI | P(A better) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['task']} | {r['question']} | {label(r['A'])} | {label(r['B'])} | {r['split']} | "
                  f"{r['diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p_a_better']:.3f} |")
    md.append("")
    md.append(f"Paired bootstrap over rows ({n_boot} resamples), metrics averaged over seeds. "
              "AUROC difference for classification; NMAE difference (sign flipped) for regression.")
    (out / "bootstrap.md").write_text("\n".join(md) + "\n")
    return df
