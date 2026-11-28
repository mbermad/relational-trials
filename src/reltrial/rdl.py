"""Relational Deep Learning baseline: heterogeneous GraphSAGE over the rel-trial graph.

Adapted from RelBench's reference implementation (examples/gnn_entity.py and
examples/model.py, Apache-2.0, https://github.com/stanford-star/relbench), restructured
so that the graph is built once and reused across seeds, and every run is logged through
the shared evaluation harness.

Pipeline:
1. ``make_pkey_fkey_graph`` turns the database into a typed graph: one node type per
   table, one edge type per primary-foreign key link, column values encoded by
   PyTorch Frame, a timestamp on every node that has one.
2. ``NeighborLoader`` with ``time_attr="time"`` samples, for each trial and seed time t,
   only neighbors whose timestamp is <= t (no leakage from the future).
3. ``Model`` encodes columns, adds a relative-time encoding, runs heterogeneous GraphSAGE,
   and reads out the trial node.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP
from torch_geometric.seed import seed_everything

from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.utils import get_stype_proposal

from .data import TaskBundle
from .evaluate import PRIMARY, RunRecord, compute_metrics
from .text import make_text_embedder


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class RDLModel(torch.nn.Module):
    """Column encoder + temporal encoder + heterogeneous GraphSAGE + MLP head."""

    def __init__(self, data: HeteroData, col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
                 num_layers: int = 2, channels: int = 128, out_channels: int = 1,
                 aggr: str = "sum", norm: str = "batch_norm", hist_dim: int = 0,
                 arch: str = "sage", num_heads: int = 1, simplified_mp: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.arch = arch
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels)
        if arch == "sage":
            self.gnn = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types,
                                       channels=channels, aggr=aggr, num_layers=num_layers)
        elif arch == "relgnn":
            from .relgnn.atomic_routes import get_atomic_routes
            from .relgnn.nn import RelGNN

            self.gnn = RelGNN(node_types=data.node_types,
                              edge_types=get_atomic_routes(data.edge_types), channels=channels,
                              aggr=aggr, num_model_layers=num_layers, num_heads=num_heads,
                              simplified_MP=simplified_mp)
        else:
            raise ValueError(f"unknown architecture {arch!r}")
        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=1)
        # Optional: time-safe label history of the trial's linked entities (see
        # history_features), encoded and added to the trial's embedding before the head.
        self.hist_enc = (MLP(hist_dim, hidden_channels=channels, out_channels=channels,
                             num_layers=2, norm=None) if hist_dim else None)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        if self.hist_enc is not None:
            self.hist_enc.reset_parameters()

    def embed(self, batch: HeteroData, entity_table: str) -> Tensor:
        """Final-layer embedding of the seed (trial) nodes; reused later for C3."""
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rt in rel_time.items():
            x_dict[nt] = x_dict[nt] + rt
        if self.dropout:
            x_dict = {k: F.dropout(v, self.dropout, self.training) for k, v in x_dict.items()}
        if self.arch == "sage":
            x_dict = self.gnn(x_dict, batch.edge_index_dict, batch.num_sampled_nodes_dict,
                              batch.num_sampled_edges_dict)
        else:  # RelGNN needs no per-hop counts (bidirectional subgraphs do not record them)
            x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return x_dict[entity_table][: seed_time.size(0)]

    def forward(self, batch: HeteroData, entity_table: str) -> Tensor:
        h = self.embed(batch, entity_table)
        if self.hist_enc is not None:
            h = h + self.hist_enc(batch[entity_table].hist)
        if self.dropout:
            h = F.dropout(h, self.dropout, self.training)
        return self.head(h)


# --------------------------------------------------------------------------------------
# Graph construction (once per task; cached on disk by PyTorch Frame)
# --------------------------------------------------------------------------------------
def build_graph(bundle: TaskBundle, cache_dir: str, text: str = "glove",
                device: Optional[str] = None, graph_history: bool = False):
    """Build the relational entity graph for a task.

    ``text='none'`` drops text columns entirely (fastest, lowest RAM); ``'hash'`` is an
    offline embedder used by the tests; ``'glove'`` matches RelBench's reference.

    ``graph_history=True`` (B) writes time-safe label history onto every junction row
    (``features.junction_history_columns``), so message passing itself can carry a sponsor's
    or condition's track record, instead of adding history at the prediction head.
    """
    db = bundle.dataset.get_db()  # dataset-level db; the task's hidden columns are removed below
    stype_path = Path(cache_dir) / "stypes.json"
    if stype_path.exists():
        col_to_stype_dict = {t: {c: stype(s) for c, s in cols.items()}
                             for t, cols in json.loads(stype_path.read_text()).items()}
    else:
        col_to_stype_dict = get_stype_proposal(db)
        stype_path.parent.mkdir(parents=True, exist_ok=True)
        stype_path.write_text(json.dumps(col_to_stype_dict, indent=2, default=str))

    if graph_history:
        from relbench.base import Database, Table

        from .features import junction_history_columns

        t = bundle.task
        extra = junction_history_columns(db, t.entity_table, t.entity_col, bundle.history_rows,
                                         t.time_col, t.target_col, t.timedelta,
                                         prior=float(bundle.train.df[t.target_col].mean()))
        tables = dict(db.table_dict)
        col_to_stype_dict = {k: dict(v) for k, v in col_to_stype_dict.items()}
        for name, cols in extra.items():
            old = tables[name]
            tables[name] = Table(old.df.assign(**{c: cols[c].to_numpy() for c in cols}),
                                 old.fkey_col_to_pkey_table, old.pkey_col, old.time_col)
            for c in cols:
                col_to_stype_dict[name][c] = stype.numerical
        db = Database(tables)
        print(f"  history in the graph: {', '.join(f'{n}.{c}' for n, cs in extra.items() for c in cs)}")

    text_cfg = None
    if text == "none":
        for cols in col_to_stype_dict.values():
            for c in [c for c, s in cols.items() if s == stype.text_embedded]:
                cols.pop(c)
    else:
        text_cfg = TextEmbedderConfig(text_embedder=make_text_embedder(text, device=device),
                                      batch_size=256)

    materialized = Path(cache_dir) / (f"materialized_{bundle.task_name}_{text}"
                                      + ("_ghist" if graph_history else ""))

    def build():
        return make_pkey_fkey_graph(
            db,
            col_to_stype_dict=col_to_stype_dict,
            text_embedder_cfg=text_cfg,
            cache_dir=str(materialized),
            remove_columns=bundle.task.hidden_columns(),
        )

    try:
        data, col_stats_dict = build()
    except (RuntimeError, EOFError, OSError) as err:
        # A run stopped while saving leaves a truncated cache file ("failed finding central
        # directory"). Throw the task's cache away and rebuild it once.
        print(f"  cache at {materialized} is unreadable ({type(err).__name__}); rebuilding it")
        import shutil

        shutil.rmtree(materialized, ignore_errors=True)
        data, col_stats_dict = build()
    return data, col_stats_dict


class AttachHistory:
    """Loader transform: after RelBench's target transform, attach each seed row's history."""

    def __init__(self, base, entity_table: str, hist: Tensor):
        self.base, self.entity_table, self.hist = base, entity_table, hist

    def __call__(self, batch: HeteroData) -> HeteroData:
        batch = self.base(batch) if self.base is not None else batch
        batch[self.entity_table].hist = self.hist[batch[self.entity_table].input_id]
        return batch


def make_loaders(bundle: TaskBundle, data: HeteroData, num_layers: int, num_neighbors: int,
                 batch_size: int, temporal_strategy: str = "uniform",
                 num_workers: int = 0,
                 hist: Optional[Dict[str, np.ndarray]] = None,
                 subgraph_type: str = "directional") -> Dict[str, NeighborLoader]:
    loaders = {}
    for split in ("train", "val", "test", "train_eval"):
        base_split = "train" if split == "train_eval" else split
        table = bundle.split(base_split)
        table_input = get_node_train_table_input(table=table, task=bundle.task)
        transform = table_input.transform
        if hist is not None:
            transform = AttachHistory(transform, bundle.task.entity_table,
                                      torch.as_tensor(hist[base_split], dtype=torch.float32))
        loaders[split] = NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(num_layers)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=transform,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            subgraph_type=subgraph_type,
            shuffle=split == "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
    return loaders


# Per-task settings from RelGNN's official code (examples/utils.py, get_configs).
RELGNN_CONFIGS = {
    ("rel-trial", "study-outcome"): dict(model_layers=2, channels=256, aggr="sum", num_heads=1,
                                         batch_size=512, num_neighbors=128, num_layers=2,
                                         subgraph_type="bidirectional"),
    ("rel-trial", "study-adverse"): dict(model_layers=1, channels=128, aggr="mean", num_heads=2,
                                         batch_size=128, num_neighbors=64, num_layers=2,
                                         subgraph_type="directional"),
    ("rel-f1", "driver-dnf"): dict(model_layers=2, channels=128, aggr="sum", num_heads=1,
                                   batch_size=256, num_neighbors=128, num_layers=2,
                                   subgraph_type="directional"),
    ("rel-f1", "driver-top3"): dict(model_layers=1, channels=128, aggr="sum", num_heads=2,
                                    simplified_mp=True, batch_size=256, num_neighbors=128,
                                    num_layers=2, subgraph_type="directional"),
    ("rel-f1", "driver-position"): dict(model_layers=1, channels=128, aggr="sum", num_heads=4,
                                        batch_size=512, num_neighbors=128, num_layers=2,
                                        subgraph_type="bidirectional"),
}


def relgnn_config(dataset_name: str, task_name: str) -> Dict[str, Any]:
    key = (Path(dataset_name).name.replace("-synthetic", ""), task_name)
    return dict(RELGNN_CONFIGS.get(key, dict(model_layers=2, channels=128, aggr="sum",
                                             num_heads=1, subgraph_type="bidirectional")))


TUNE_SPACE = {
    "lr": [0.001, 0.002, 0.005, 0.01],
    "channels": [64, 128, 256],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "num_neighbors": [32, 64, 128],
}


def tune(bundle: TaskBundle, data: HeteroData, col_stats_dict, n_trials: int = 12,
         seed: int = 0, base: Optional[Dict[str, Any]] = None, **fixed) -> Dict[str, Any]:
    """Random search on validation with seed 0 and early stopping; returns the best params.

    Same protocol as LightGBM (fixed budget, scored on validation); trials are not logged
    as runs. ``base`` is always tried first, so tuning can only match or beat it.
    """
    metric, higher = PRIMARY[bundle.task.task_type.value]
    rng = np.random.default_rng(seed)
    trials = [dict(base or {})]
    while len(trials) < n_trials:
        trials.append({k: v[int(rng.integers(len(v)))] for k, v in TUNE_SPACE.items()})
    best, best_score = None, None
    for i, params in enumerate(trials):
        kw = {**fixed, **params}
        rec = run(bundle, data, col_stats_dict, seeds=[seed], save=False, save_preds=None, **kw)[0]
        score = rec.params["best_val"]
        if best is None or (score > best_score if higher else score < best_score):
            best, best_score = params, score
        print(f"  tune {i + 1:02d}/{n_trials} {params} -> val {metric} {score:.4f} (best {best_score:.4f})")
    return best


def history_features(bundle: TaskBundle) -> Dict[str, np.ndarray]:
    """Time-safe label history of each seed row's linked entities, as a dense matrix.

    Reuses the tested LightGBM feature code (``features.FeatureBuilder``): for every linked
    sponsor / condition / facility / intervention, the smoothed mean outcome of its earlier
    trials and how many there were, counting a label only once it is known (label time +
    the task's window <= seed time). Missing values -> the training prior (rates) or 0
    (counts); counts are log1p-scaled; everything is standardized with train statistics.
    """
    from .lgbm import build_features

    X, _ = build_features(bundle, "agg")
    cols = [c for c in X["train"].columns if "_hist_" in c]
    if not cols:
        raise ValueError("no label-history features found for this task")
    prior = float(bundle.train.df[bundle.task.target_col].mean())
    out = {}
    for split, df in X.items():
        m = df[cols].astype(float).copy()
        for c in cols:
            if c.endswith("_n_sum"):
                m[c] = np.log1p(m[c].fillna(0.0))
            else:
                m[c] = m[c].fillna(prior)
        out[split] = m.to_numpy(dtype=np.float32)
    mu, sd = out["train"].mean(0), out["train"].std(0) + 1e-6
    out = {k: (v - mu) / sd for k, v in out.items()}
    bad = [c for i, c in enumerate(cols) if not all(np.isfinite(v[:, i]).all() for v in out.values())]
    if bad:
        raise ValueError(f"non-finite label-history features: {bad}")
    out["columns"] = np.array(cols)
    print(f"  label-history features for the GNN: {len(cols)} ({', '.join(cols)})")
    return out


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def run(bundle: TaskBundle, data: HeteroData, col_stats_dict, seeds: List[int] = (0, 1, 2),
        epochs: int = 10, lr: float = 0.005, batch_size: int = 512, channels: int = 128,
        num_layers: int = 2, num_neighbors: int = 128, aggr: str = "sum",
        max_steps_per_epoch: int = 2000, out_dir: str = "results/runs",
        save_preds: Optional[str] = "results/preds", model_name: str = "rdl_sage",
        save_embeddings: Optional[str] = None,
        hist: Optional[Dict[str, np.ndarray]] = None, arch: str = "sage",
        model_layers: Optional[int] = None, num_heads: int = 1, simplified_mp: bool = False,
        subgraph_type: str = "directional", dropout: float = 0.0, weight_decay: float = 0.0,
        patience: Optional[int] = None, save: bool = True,
        mask_eval: bool = False) -> List[RunRecord]:
    """Train ``len(seeds)`` models and log each. ``num_layers`` is the number of sampled hops;
    ``model_layers`` (default: the same) is the number of message-passing layers."""
    model_layers = model_layers or num_layers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = bundle.task
    task_type = task.task_type.value
    metric, higher = PRIMARY[task_type]
    entity_table = task.entity_table
    loaders = make_loaders(bundle, data, num_layers, num_neighbors, batch_size, hist=hist,
                           subgraph_type=subgraph_type)
    hist_dim = 0 if hist is None else int(hist["train"].shape[1])
    y = {s: bundle.split(s).df[task.target_col].to_numpy(dtype=float) for s in ("val", "test")}

    if bundle.is_classification:
        loss_fn, clamp = BCEWithLogitsLoss(), None
    else:
        loss_fn = L1Loss()
        clamp = np.percentile(bundle.train.df[task.target_col].to_numpy(), [2, 98])

    @torch.no_grad()
    def predict(model, loader) -> np.ndarray:
        model.eval()
        out = []
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch, entity_table).view(-1)
            if clamp is not None:
                pred = torch.clamp(pred, float(clamp[0]), float(clamp[1]))
            else:
                pred = torch.sigmoid(pred)
            out.append(pred.cpu())
        return torch.cat(out).numpy()

    records = []
    for seed in seeds:
        seed_everything(seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = RDLModel(data, col_stats_dict, num_layers=model_layers, channels=channels,
                         aggr=aggr, hist_dim=hist_dim, arch=arch, num_heads=num_heads,
                         simplified_mp=simplified_mp, dropout=dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        since_best = 0
        best_state, best_score, history = None, (-math.inf if higher else math.inf), []
        t0 = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = n_sum = steps = 0
            for batch in loaders["train"]:
                batch = batch.to(device)
                opt.zero_grad()
                pred = model(batch, entity_table).view(-1)
                loss = loss_fn(pred.float(), batch[entity_table].y.float())
                loss.backward()
                opt.step()
                loss_sum += loss.item() * pred.numel()
                n_sum += pred.numel()
                steps += 1
                if steps >= max_steps_per_epoch:
                    break
            val_pred = predict(model, loaders["val"])
            score = compute_metrics(task_type, y["val"], val_pred, bundle.nmae_std())[metric]
            history.append({"epoch": epoch, "train_loss": loss_sum / max(n_sum, 1), f"val_{metric}": score})
            print(f"  seed {seed} epoch {epoch:02d}: loss {loss_sum / max(n_sum, 1):.4f}  val {metric} {score:.4f}")
            if (score >= best_score) if higher else (score <= best_score):
                best_score, best_state = score, copy.deepcopy(model.state_dict())
                since_best = 0
            else:
                since_best += 1
                if patience is not None and since_best >= patience:
                    print(f"  early stop at epoch {epoch} (no val improvement for {patience})")
                    break
        train_seconds = time.perf_counter() - t0

        model.load_state_dict(best_state)
        preds = {"val": predict(model, loaders["val"])}
        t1 = time.perf_counter()
        preds["test"] = predict(model, loaders["test"])
        infer = (time.perf_counter() - t1) / max(len(preds["test"]), 1) * 1000
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None

        rec = RunRecord(
            task=bundle.task_name, model=model_name, seed=seed, task_type=task_type,
            val=compute_metrics(task_type, y["val"], preds["val"], bundle.nmae_std()),
            test=compute_metrics(task_type, y["test"], preds["test"], bundle.nmae_std()),
            train_seconds=train_seconds, infer_seconds_per_1k=infer, peak_gpu_mem_mb=peak,
            params={"epochs": epochs, "lr": lr, "batch_size": batch_size, "channels": channels,
                    "num_layers": num_layers, "num_neighbors": num_neighbors, "aggr": aggr,
                    "arch": arch, "model_layers": model_layers, "num_heads": num_heads,
                    "simplified_mp": simplified_mp, "subgraph_type": subgraph_type,
                    "dropout": dropout, "weight_decay": weight_decay, "patience": patience,
                    "best_val": best_score,
                    "max_steps_per_epoch": max_steps_per_epoch, "device": device.type,
                    "label_history_features": hist_dim, "history": history},
        )
        if save:
            rec.save(out_dir)
        if save and save_preds:
            d = Path(save_preds) / bundle.task_name / model_name
            d.mkdir(parents=True, exist_ok=True)
            for s, p in preds.items():
                np.save(d / f"seed{seed}_{s}.npy", p)
        if save and mask_eval and hist is not None:
            _mask_eval(model, loaders, hist, predict, bundle, y, rec, Path(out_dir).parent,
                       save_preds)
        if save and save_embeddings and seed == seeds[0]:
            export_embeddings(model, loaders, entity_table, device,
                              Path(save_embeddings) / bundle.task_name / model_name)
        print(f"  seed {seed}: val {rec.val}  test {rec.test}  ({train_seconds:.0f}s)")
        records.append(rec)
    return records


def _mask_eval(model, loaders, hist, predict, bundle, y, rec, root: Path, save_preds) -> None:
    """Deletion test: zero the standardized history columns (= training mean) at inference."""
    from . import masking

    groups = masking.history_groups([str(c) for c in hist["columns"]])

    def run_masked(split, idx):
        tr = loaders[split].transform
        original = tr.hist
        masked = original.clone()
        masked[:, idx] = 0.0
        tr.hist = masked
        try:
            return predict(model, loaders[split])
        finally:
            tr.hist = original

    res = masking.evaluate_masked(run_masked, groups, y, bundle.task.task_type.value,
                                  bundle.nmae_std())
    masking.save(res, root / "masking", bundle.task_name, rec.model, rec.seed, save_preds,
                 unmasked={"val": rec.val, "test": rec.test})


@torch.no_grad()
def export_embeddings(model: RDLModel, loaders, entity_table: str, device, out: Path) -> None:
    """Batch job for C3: save each split's trial embeddings (in label-table row order)."""
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    for split in ("train", "val", "test"):
        loader = loaders["train_eval" if split == "train" else split]  # unshuffled order
        embs = [model.embed(b.to(device), entity_table).cpu() for b in loader]
        np.save(out / f"{split}.npy", torch.cat(embs).numpy())
