"""LightGBM baselines: a fixed tuning budget on validation, then several seeds."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import TaskBundle
from .evaluate import PRIMARY, RunRecord, Timer, compute_metrics
from .features import FeatureBuilder

SEARCH_SPACE = {
    "num_leaves": ("int", 15, 255),
    "learning_rate": ("log", 0.01, 0.1),
    "min_child_samples": ("int", 5, 100),
    "feature_fraction": ("float", 0.5, 1.0),
    "bagging_fraction": ("float", 0.5, 1.0),
    "lambda_l2": ("log", 1e-3, 10.0),
}


def sample_params(rng: np.random.Generator) -> Dict:
    p = {}
    for name, (kind, lo, hi) in SEARCH_SPACE.items():
        if kind == "int":
            p[name] = int(rng.integers(lo, hi + 1))
        elif kind == "log":
            p[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            p[name] = float(rng.uniform(lo, hi))
    return p


def build_features(bundle: TaskBundle, feature_set: str, as_of: str = "seed"):
    """Feature sets: 'table' (trial row only), 'agg' (+ relational aggregates and label
    history), 'agg_nohist' ('agg' without any label-history feature: the ablation).

    ``as_of='start'`` computes every feature as of the entity's own start date instead of the
    seed time (a control for B: the history written into the graph is as of each link's
    date, i.e. the trial's start, so it knows fewer outcomes than seed-time history).
    """
    task = bundle.task
    history = None
    if feature_set == "agg":
        # Train + val labels, used only once known (label time + timedelta <= seed time).
        history = bundle.history_rows
    fb = FeatureBuilder(bundle.db, task.entity_table, task.entity_col, task.time_col,
                        task.target_col, task.timedelta,
                        feature_set="table" if feature_set == "table" else "agg",
                        label_history=history,
                        # Activity counts are new with rel-stack; rel-trial / rel-f1 keep
                        # the exact features their published numbers were made with.
                        activity=bundle.base_name not in ("rel-trial", "rel-f1"))
    frames = {s: bundle.split(s).df for s in ("train", "val", "test")}
    if as_of == "start":
        ent = bundle.db.table_dict[task.entity_table]
        if ent.time_col is None:
            raise ValueError(f"{task.entity_table} has no time column: as_of='start' needs one")
        start = ent.df.set_index(ent.pkey_col)[ent.time_col]
        frames = {s: df.assign(**{task.time_col: np.minimum(
            pd.to_datetime(df[task.entity_col].map(start)).fillna(df[task.time_col]).to_numpy(),
            pd.to_datetime(df[task.time_col]).to_numpy())}) for s, df in frames.items()}
    elif as_of != "seed":
        raise ValueError("as_of must be 'seed' or 'start'")
    fb.fit(frames["train"])
    X = {s: _clean_names(fb.transform(df)) for s, df in frames.items()}
    y = {s: bundle.split(s).df[task.target_col].to_numpy(dtype=float) for s in X}
    return X, y


def _clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """LightGBM rejects some characters in feature names."""
    import re

    return df.rename(columns=lambda c: re.sub(r"[^0-9A-Za-z_]+", "__", str(c)))


def _lgb_params(bundle: TaskBundle, params: Dict, seed: int) -> Dict:
    base = {
        "objective": "binary" if bundle.is_classification else "l1",
        "metric": "auc" if bundle.is_classification else "l1",
        "verbosity": -1,
        "bagging_freq": 1,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "deterministic": True,
        "num_threads": 0,
    }
    return {**base, **params}


def fit_predict(bundle: TaskBundle, X, y, params: Dict, seed: int,
                max_rounds: int = 2000, patience: int = 100):
    dtrain = lgb.Dataset(X["train"], y["train"], free_raw_data=False)
    dval = lgb.Dataset(X["val"], y["val"], reference=dtrain, free_raw_data=False)
    with Timer() as timer:
        booster = lgb.train(_lgb_params(bundle, params, seed), dtrain, num_boost_round=max_rounds,
                            valid_sets=[dval], callbacks=[lgb.early_stopping(patience, verbose=False)])
    preds = {}
    for s in ("val", "test"):
        t0 = time.perf_counter()
        p = booster.predict(X[s], num_iteration=booster.best_iteration)
        if s == "test":
            infer = (time.perf_counter() - t0) / max(len(p), 1) * 1000
        preds[s] = np.clip(p, 0, None) if bundle.is_regression else p
    return booster, preds, timer.seconds, infer


def tune(bundle: TaskBundle, X, y, n_trials: int, seed: int = 0) -> Dict:
    """Random search with a fixed budget, scored on validation."""
    task_type = bundle.task.task_type.value
    metric, higher = PRIMARY[task_type]
    rng = np.random.default_rng(seed)
    best, best_score = None, None
    for i in range(n_trials):
        params = sample_params(rng)
        _, preds, _, _ = fit_predict(bundle, X, y, params, seed)
        score = compute_metrics(task_type, y["val"], preds["val"], bundle.nmae_std())[metric]
        if best is None or (score > best_score if higher else score < best_score):
            best, best_score = params, score
        print(f"  trial {i + 1:02d}/{n_trials}: val {metric}={score:.4f} (best {best_score:.4f})")
    return best


def run(bundle: TaskBundle, feature_set: str = "table", seeds: List[int] = (0, 1, 2),
        n_trials: int = 20, out_dir: str = "results/runs",
        save_preds: Optional[str] = "results/preds", mask_eval: bool = False) -> List[RunRecord]:
    model_name = f"lgbm_{feature_set}"
    task_type = bundle.task.task_type.value
    print(f"[{bundle.task_name}] building '{feature_set}' features ...")
    with Timer() as ft:
        X, y = build_features(bundle, feature_set)
    print(f"  {X['train'].shape[1]} features, {len(y['train'])} train rows "
          f"({ft.seconds:.1f}s)")
    print(f"[{bundle.task_name}] tuning {model_name} ({n_trials} trials) ...")
    best = tune(bundle, X, y, n_trials)
    records = []
    for seed in seeds:
        booster, preds, secs, infer = fit_predict(bundle, X, y, best, seed)
        rec = RunRecord(
            task=bundle.task_name, model=model_name, seed=seed, task_type=task_type,
            val=compute_metrics(task_type, y["val"], preds["val"], bundle.nmae_std()),
            test=compute_metrics(task_type, y["test"], preds["test"], bundle.nmae_std()),
            train_seconds=secs, infer_seconds_per_1k=infer,
            params={**best, "best_iteration": booster.best_iteration,
                    "n_features": X["train"].shape[1], "n_trials": n_trials,
                    "feature_seconds": ft.seconds},
        )
        rec.save(out_dir)
        if save_preds:
            _save_preds(save_preds, rec, preds)
        _save_importance(booster, bundle.task_name, model_name, seed, Path(out_dir).parent / "importance")
        if mask_eval:
            _mask_eval(bundle, booster, X, y, rec, Path(out_dir).parent, save_preds)
        print(f"  seed {seed}: val {rec.val}  test {rec.test}")
        records.append(rec)
    return records


def _mask_eval(bundle: TaskBundle, booster, X, y, rec: RunRecord, root: Path, save_preds) -> None:
    """Deletion test: history columns -> their training mean, model unchanged."""
    from . import masking

    cols = list(X["train"].columns)
    groups = masking.history_groups(cols)
    if not groups["all"]:
        return
    means = X["train"].iloc[:, groups["all"]].astype(float).mean()

    def predict(split, idx):
        Xm = X[split].copy()
        for i in idx:
            Xm[cols[i]] = means[cols[i]]
        p = booster.predict(Xm, num_iteration=booster.best_iteration)
        return np.clip(p, 0, None) if bundle.is_regression else p

    res = masking.evaluate_masked(predict, groups, y, bundle.task.task_type.value, bundle.nmae_std())
    masking.save(res, root / "masking", bundle.task_name, rec.model, rec.seed, save_preds,
                 unmasked={"val": rec.val, "test": rec.test})


def _save_preds(root: str, rec: RunRecord, preds: Dict[str, np.ndarray]) -> None:
    from pathlib import Path

    d = Path(root) / rec.task / rec.model
    d.mkdir(parents=True, exist_ok=True)
    for s, p in preds.items():
        np.save(d / f"seed{rec.seed}_{s}.npy", p)


def _save_importance(booster, task: str, model: str, seed: int, root: Path) -> None:
    d = Path(root) / task
    d.mkdir(parents=True, exist_ok=True)
    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain")}).sort_values("gain", ascending=False)
    imp.to_csv(d / f"{model}_seed{seed}.csv", index=False)
