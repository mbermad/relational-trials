"""Blend (F, hybrid): does the GNN add anything LightGBM does not already have?

For each seed k, blend LightGBM seed k with GNN seed k: ``w * A + (1 - w) * B`` on
rank-normalized scores (classification; AUROC only depends on ranks) or raw predictions
(regression), with ``w`` picked on validation from a grid. Test is never used to pick w.
Logged like any model (runs + preds), named ``blend_<A>__<B>``, so tables and the paired
bootstrap pick it up. Cost = the sum of both models' training time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from .data import TaskBundle
from .evaluate import PRIMARY, RunRecord, compute_metrics

DEFAULT_PAIRS: Sequence[Tuple[str, str]] = (("lgbm_agg", "rdl_sage_hist"), ("lgbm_agg", "rdl_sage"),
                                            ("lgbm_agg_nohist", "rdl_sage"))
GRID = np.round(np.linspace(0, 1, 21), 2)


def _norm(p: np.ndarray, classification: bool) -> np.ndarray:
    return rankdata(p) / len(p) if classification else p


def blend_name(a: str, b: str) -> str:
    return f"blend_{a}__{b}"


def run(bundles: Dict[str, TaskBundle], results: str, pairs=DEFAULT_PAIRS) -> List[RunRecord]:
    root = Path(results)
    records = []
    for task, bundle in bundles.items():
        task_type = bundle.task.task_type.value
        metric, higher = PRIMARY[task_type]
        cls = bundle.is_classification
        y = {s: bundle.split(s).df[bundle.task.target_col].to_numpy(dtype=float) for s in ("val", "test")}
        for a, b in pairs:
            pa, pb = root / "preds" / task / a, root / "preds" / task / b
            seeds = sorted({int(f.name[4:].split("_")[0]) for f in pa.glob("seed*_val.npy")}
                           & {int(f.name[4:].split("_")[0]) for f in pb.glob("seed*_val.npy")})
            if not seeds:
                print(f"  [{task}] skip blend {a} + {b}: predictions missing")
                continue
            name = blend_name(a, b)
            for seed in seeds:
                P = {s: (_norm(np.load(pa / f"seed{seed}_{s}.npy"), cls),
                         _norm(np.load(pb / f"seed{seed}_{s}.npy"), cls)) for s in ("val", "test")}

                def score(w, s):
                    p = w * P[s][0] + (1 - w) * P[s][1]
                    return compute_metrics(task_type, y[s], p, bundle.nmae_std())[metric]

                vals = [score(w, "val") for w in GRID]
                w = float(GRID[int(np.argmax(vals) if higher else np.argmin(vals))])
                preds = {s: w * P[s][0] + (1 - w) * P[s][1] for s in ("val", "test")}
                cost = 0.0
                for m in (a, b):
                    f = root / "runs" / task / m / f"seed{seed}.json"
                    if f.exists():
                        r = json.loads(f.read_text())
                        cost += r["train_seconds"] + r.get("params", {}).get("feature_seconds", 0.0)
                rec = RunRecord(task=task, model=name, seed=seed, task_type=task_type,
                                val=compute_metrics(task_type, y["val"], preds["val"], bundle.nmae_std()),
                                test=compute_metrics(task_type, y["test"], preds["test"], bundle.nmae_std()),
                                train_seconds=cost, params={"weight_on_A": w, "A": a, "B": b},
                                notes="weight picked on validation")
                rec.save(root / "runs")
                d = root / "preds" / task / name
                d.mkdir(parents=True, exist_ok=True)
                for s, p in preds.items():
                    np.save(d / f"seed{seed}_{s}.npy", p)
                print(f"  [{task}] {name} seed {seed}: w={w:.2f} val {rec.val} test {rec.test}")
                records.append(rec)
    return records
