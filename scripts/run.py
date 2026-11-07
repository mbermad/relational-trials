"""One entry point for every milestone result.

Examples:
    python scripts/run.py explore
    python scripts/run.py lgbm --features table --tasks study-outcome study-adverse
    python scripts/run.py lgbm --features agg
    python scripts/run.py rdl --text glove
    python scripts/run.py tables
    python scripts/run.py all            # everything above, in order

Add ``--dataset path/to/local/folder`` to run on a local copy (the tests use a tiny
synthetic one). Results land in ``results/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reltrial import data as D  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["explore", "lgbm", "rdl", "tables", "all"])
    p.add_argument("--dataset", default=D.DEFAULT_DATASET)
    p.add_argument("--tasks", nargs="+", default=list(D.TASKS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--results", default="results")
    # LightGBM
    p.add_argument("--features", nargs="+", default=["table", "agg"], choices=["table", "agg"])
    p.add_argument("--n_trials", type=int, default=20, help="tuning budget per model")
    # RDL
    p.add_argument("--text", default="glove", choices=["glove", "hash", "none"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_neighbors", type=int, default=128)
    p.add_argument("--max_steps_per_epoch", type=int, default=2000)
    p.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/reltrial"))
    p.add_argument("--save_embeddings", action="store_true",
                   help="export trial embeddings for C3 (first seed only)")
    return p.parse_args()


def main() -> None:
    a = parse()
    runs = f"{a.results}/runs"
    preds = f"{a.results}/preds"
    cmds = ["explore", "lgbm", "rdl", "tables"] if a.command == "all" else [a.command]
    dataset = None
    bundles = {}

    def bundle(task):
        nonlocal dataset
        if task not in bundles:
            print(f"Loading {a.dataset} / {task} ...")
            bundles[task] = D.load_bundle(task, a.dataset, dataset=dataset)
            dataset = bundles[task].dataset
        return bundles[task]

    for cmd in cmds:
        if cmd == "explore":
            from reltrial import explore

            explore.run({t: bundle(t) for t in a.tasks}, out_dir=f"{a.results}/explore")
        elif cmd == "lgbm":
            from reltrial import lgbm

            for t in a.tasks:
                for fs in a.features:
                    lgbm.run(bundle(t), feature_set=fs, seeds=a.seeds, n_trials=a.n_trials,
                             out_dir=runs, save_preds=preds)
        elif cmd == "rdl":
            from reltrial import rdl

            for t in a.tasks:
                b = bundle(t)
                print(f"[{t}] building graph (text={a.text}) ...")
                data, stats = rdl.build_graph(b, cache_dir=f"{a.cache_dir}/{Path(a.dataset).name}",
                                              text=a.text)
                rdl.run(b, data, stats, seeds=a.seeds, epochs=a.epochs, lr=a.lr,
                        batch_size=a.batch_size, channels=a.channels, num_layers=a.num_layers,
                        num_neighbors=a.num_neighbors, max_steps_per_epoch=a.max_steps_per_epoch,
                        out_dir=runs, save_preds=preds,
                        save_embeddings=f"{a.results}/embeddings" if a.save_embeddings else None)
        elif cmd == "tables":
            from reltrial import tables

            tables.build(runs_dir=runs, out_dir=f"{a.results}/tables")


if __name__ == "__main__":
    main()
