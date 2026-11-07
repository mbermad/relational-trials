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
                 aggr: str = "sum", norm: str = "batch_norm"):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={nt: data[nt].tf.col_names_dict for nt in data.node_types},
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]], channels=channels)
        self.gnn = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types,
                                   channels=channels, aggr=aggr, num_layers=num_layers)
        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=1)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def embed(self, batch: HeteroData, entity_table: str) -> Tensor:
        """Final-layer embedding of the seed (trial) nodes; reused later for C3."""
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        rel_time = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for nt, rt in rel_time.items():
            x_dict[nt] = x_dict[nt] + rt
        x_dict = self.gnn(x_dict, batch.edge_index_dict, batch.num_sampled_nodes_dict,
                          batch.num_sampled_edges_dict)
        return x_dict[entity_table][: seed_time.size(0)]

    def forward(self, batch: HeteroData, entity_table: str) -> Tensor:
        return self.head(self.embed(batch, entity_table))


# --------------------------------------------------------------------------------------
# Graph construction (once per task; cached on disk by PyTorch Frame)
# --------------------------------------------------------------------------------------
def build_graph(bundle: TaskBundle, cache_dir: str, text: str = "glove",
                device: Optional[str] = None):
    """Build the relational entity graph for a task.

    ``text='none'`` drops text columns entirely (fastest, lowest RAM); ``'hash'`` is an
    offline embedder used by the tests; ``'glove'`` matches RelBench's reference.
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

    text_cfg = None
    if text == "none":
        for cols in col_to_stype_dict.values():
            for c in [c for c, s in cols.items() if s == stype.text_embedded]:
                cols.pop(c)
    else:
        text_cfg = TextEmbedderConfig(text_embedder=make_text_embedder(text, device=device),
                                      batch_size=256)

    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_cfg,
        cache_dir=str(Path(cache_dir) / f"materialized_{bundle.task_name}_{text}"),
        remove_columns=bundle.task.hidden_columns(),
    )
    return data, col_stats_dict


def make_loaders(bundle: TaskBundle, data: HeteroData, num_layers: int, num_neighbors: int,
                 batch_size: int, temporal_strategy: str = "uniform",
                 num_workers: int = 0) -> Dict[str, NeighborLoader]:
    loaders = {}
    for split in ("train", "val", "test", "train_eval"):
        table = bundle.split("train" if split == "train_eval" else split)
        table_input = get_node_train_table_input(table=table, task=bundle.task)
        loaders[split] = NeighborLoader(
            data,
            num_neighbors=[int(num_neighbors / 2**i) for i in range(num_layers)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=batch_size,
            temporal_strategy=temporal_strategy,
            shuffle=split == "train",
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
    return loaders


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def run(bundle: TaskBundle, data: HeteroData, col_stats_dict, seeds: List[int] = (0, 1, 2),
        epochs: int = 10, lr: float = 0.005, batch_size: int = 512, channels: int = 128,
        num_layers: int = 2, num_neighbors: int = 128, aggr: str = "sum",
        max_steps_per_epoch: int = 2000, out_dir: str = "results/runs",
        save_preds: Optional[str] = "results/preds", model_name: str = "rdl_sage",
        save_embeddings: Optional[str] = None) -> List[RunRecord]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = bundle.task
    task_type = task.task_type.value
    metric, higher = PRIMARY[task_type]
    entity_table = task.entity_table
    loaders = make_loaders(bundle, data, num_layers, num_neighbors, batch_size)
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
        model = RDLModel(data, col_stats_dict, num_layers=num_layers, channels=channels,
                         aggr=aggr).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
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
                    "max_steps_per_epoch": max_steps_per_epoch, "device": device.type,
                    "history": history},
        )
        rec.save(out_dir)
        if save_preds:
            d = Path(save_preds) / bundle.task_name / model_name
            d.mkdir(parents=True, exist_ok=True)
            for s, p in preds.items():
                np.save(d / f"seed{seed}_{s}.npy", p)
        if save_embeddings and seed == seeds[0]:
            export_embeddings(model, loaders, entity_table, device,
                              Path(save_embeddings) / bundle.task_name / model_name)
        print(f"  seed {seed}: val {rec.val}  test {rec.test}  ({train_seconds:.0f}s)")
        records.append(rec)
    return records


@torch.no_grad()
def export_embeddings(model: RDLModel, loaders, entity_table: str, device, out: Path) -> None:
    """Batch job for C3: save each split's trial embeddings (in label-table row order)."""
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    for split in ("train", "val", "test"):
        loader = loaders["train_eval" if split == "train" else split]  # unshuffled order
        embs = [model.embed(b.to(device), entity_table).cpu() for b in loader]
        np.save(out / f"{split}.npy", torch.cat(embs).numpy())
