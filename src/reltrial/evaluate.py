"""Shared evaluation harness: metrics, run logging, multi-seed aggregation, tables.

Every model writes one JSON file per (task, model, seed) to ``results/runs``. The table
builder reads those files, so models never need to know about each other.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score

# Primary metric per task type, and whether higher is better.
PRIMARY = {"binary_classification": ("roc_auc", True), "regression": ("nmae", False)}

# Test-set numbers published on the RelBench v3 leaderboard (single run each), for the
# "do we reproduce the reference?" sanity check. Regression is NMAE.
# Source: github.com/stanford-star/relbench, leaderboard/entries/{390,380}.json
REFERENCE = {
    ("study-outcome", "lgbm_table"): 0.6885,
    ("study-outcome", "rdl_sage"): 0.6693,
    ("study-adverse", "lgbm_table"): 0.1293,
    ("study-adverse", "rdl_sage"): 0.1293,
}


def compute_metrics(task_type: str, y_true: np.ndarray, y_pred: np.ndarray,
                    nmae_std: Optional[float] = None) -> Dict[str, float]:
    """AUROC for classification; MAE and NMAE (= MAE / train std) for regression."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} labels vs {len(y_pred)} preds")
    if task_type == "binary_classification":
        return {"roc_auc": float(roc_auc_score(y_true, y_pred))}
    if task_type == "regression":
        mae = float(mean_absolute_error(y_true, y_pred))
        out = {"mae": mae}
        if nmae_std:
            out["nmae"] = mae / float(nmae_std)
        return out
    raise ValueError(f"unsupported task type {task_type}")


@dataclass
class RunRecord:
    task: str
    model: str
    seed: int
    task_type: str
    val: Dict[str, float]
    test: Dict[str, float]
    train_seconds: float
    infer_seconds_per_1k: Optional[float] = None
    peak_gpu_mem_mb: Optional[float] = None
    params: Dict = field(default_factory=dict)
    notes: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    host: str = field(default_factory=platform.node)

    def save(self, root: str | Path = "results/runs") -> Path:
        path = Path(root) / self.task / self.model / f"seed{self.seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


def load_runs(root: str | Path = "results/runs") -> List[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(root).glob("*/*/seed*.json"))]


def mean_ci(values: Iterable[float], level: float = 0.95) -> tuple[float, float, int]:
    """Mean and half-width of a t-based confidence interval over seeds."""
    v = np.asarray([x for x in values if x is not None and not math.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float("nan"), 1
    from scipy import stats  # scipy ships with scikit-learn

    half = stats.t.ppf(0.5 + level / 2, df=n - 1) * v.std(ddof=1) / math.sqrt(n)
    return float(v.mean()), float(half), n


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     metric: str, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Bootstrap the test set to estimate P(model A beats model B) on ``metric``.

    Used later (C1) to back every "A beats B" claim; included now so the harness is done.
    """
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    higher = metric == "roc_auc"
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if metric == "roc_auc":
            if len(np.unique(y_true[idx])) < 2:
                continue
            a, b = roc_auc_score(y_true[idx], pred_a[idx]), roc_auc_score(y_true[idx], pred_b[idx])
        else:
            a = mean_absolute_error(y_true[idx], pred_a[idx])
            b = mean_absolute_error(y_true[idx], pred_b[idx])
        diffs.append(a - b if higher else b - a)  # > 0 means A is better
    diffs = np.asarray(diffs)
    return {
        "mean_improvement": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_a_better": float((diffs > 0).mean()),
    }


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
