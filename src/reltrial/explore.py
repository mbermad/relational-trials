"""Data exploration for the milestone: schema summary, label statistics, figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from relbench.base import Database  # noqa: E402

from .data import TaskBundle  # noqa: E402

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
COLOR = "#2f6db5"


def schema_summary(db: Database) -> pd.DataFrame:
    rows = []
    for name, t in sorted(db.table_dict.items()):
        df = t.df
        rng = ""
        if t.time_col is not None and len(df):
            ts = pd.to_datetime(df[t.time_col])
            rng = f"{ts.min():%Y-%m} to {ts.max():%Y-%m}"
        rows.append({"table": name, "rows": len(df), "columns": df.shape[1],
                     "primary_key": t.pkey_col or "", "time_col": t.time_col or "",
                     "foreign_keys": ", ".join(f"{c}->{p}" for c, p in t.fkey_col_to_pkey_table.items()),
                     "time_range": rng})
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


def label_summary(bundle: TaskBundle) -> pd.DataFrame:
    tgt = bundle.task.target_col
    rows = []
    for s in ("train", "val", "test"):
        df = bundle.split(s).df
        y = df[tgt].astype(float)
        ts = pd.to_datetime(df[bundle.task.time_col])
        r = {"task": bundle.task_name, "split": s, "rows": len(df),
             "seed_times": ts.nunique(), "from": f"{ts.min():%Y-%m-%d}", "to": f"{ts.max():%Y-%m-%d}"}
        if bundle.is_classification:
            r["positive_rate"] = round(float(y.mean()), 4)
        else:
            r.update(mean=round(float(y.mean()), 2), median=float(y.median()),
                     p95=float(y.quantile(0.95)), share_zero=round(float((y == 0).mean()), 4))
        rows.append(r)
    return pd.DataFrame(rows)


def plot_trials_per_year(db: Database, entity_table: str, out: Path, val_ts, test_ts) -> None:
    t = db.table_dict[entity_table]
    years = pd.to_datetime(t.df[t.time_col]).dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    ax.bar(years.index, years.values, color=COLOR, width=0.8)
    for ts, lab in ((val_ts, "val"), (test_ts, "test")):
        ax.axvline(pd.Timestamp(ts).year - 0.5, color="0.3", ls="--", lw=0.8)
        ax.text(pd.Timestamp(ts).year - 0.4, years.max() * 0.95, lab, fontsize=7, color="0.3")
    ax.set_xlabel("Trial start year")
    ax.set_ylabel("Trials")
    ax.set_title("Trials in rel-trial by start year", loc="left")
    fig.tight_layout()
    fig.savefig(out / "trials_per_year.png")
    plt.close(fig)


def plot_outcome_by_year(bundle: TaskBundle, out: Path) -> None:
    frames = []
    for s in ("train", "val", "test"):
        df = bundle.split(s).df
        frames.append(pd.DataFrame({"year": pd.to_datetime(df[bundle.task.time_col]).dt.year,
                                    "y": df[bundle.task.target_col].astype(float), "split": s}))
    d = pd.concat(frames)
    g = d.groupby("year").agg(rate=("y", "mean"), n=("y", "size")).reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    ax.plot(g["year"], g["rate"], marker="o", ms=3, color=COLOR)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Seed year (label window starts)")
    ax.set_ylabel("Share meeting primary outcome")
    ax.set_title("study-outcome: positive rate by year", loc="left")
    fig.tight_layout()
    fig.savefig(out / "outcome_rate_by_year.png")
    plt.close(fig)


def plot_adverse_distribution(bundle: TaskBundle, out: Path) -> None:
    y = bundle.train.df[bundle.task.target_col].astype(float)
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    ax.hist(np.log10(1 + y), bins=40, color=COLOR)
    ax.set_xlabel("log10(1 + patients with serious adverse events or death)")
    ax.set_ylabel("Trials (train)")
    ax.set_title(f"study-adverse: heavy-tailed target (median {y.median():.0f}, "
                 f"95th pct {y.quantile(0.95):.0f})", loc="left")
    fig.tight_layout()
    fig.savefig(out / "adverse_distribution.png")
    plt.close(fig)


def plot_missingness(db: Database, entity_table: str, out: Path, top: int = 15) -> None:
    df = db.table_dict[entity_table].df
    miss = df.isna().mean().sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.barh(miss.index, miss.values, color=COLOR)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share missing")
    ax.set_title(f"Most-missing columns in '{entity_table}'", loc="left")
    fig.tight_layout()
    fig.savefig(out / "missingness.png")
    plt.close(fig)


def run(bundles: Dict[str, TaskBundle], out_dir: str = "results/explore") -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(bundles.values()))
    db = first.dataset.get_db()
    ds = first.dataset
    schema = schema_summary(db)
    schema.to_csv(out / "schema.csv", index=False)
    labels = pd.concat([label_summary(b) for b in bundles.values()], ignore_index=True)
    labels.to_csv(out / "labels.csv", index=False)

    plot_trials_per_year(db, first.task.entity_table, out, ds.val_timestamp, ds.test_timestamp)
    plot_missingness(db, first.task.entity_table, out)
    for b in bundles.values():
        if b.is_classification:
            plot_outcome_by_year(b, out)
        else:
            plot_adverse_distribution(b, out)

    summary = {
        "tables": int(len(schema)),
        "rows": int(schema["rows"].sum()),
        "columns": int(schema["columns"].sum()),
        "val_timestamp": str(ds.val_timestamp),
        "test_timestamp": str(ds.test_timestamp),
        "labels": labels.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _write_latex(schema, labels, summary, out)
    print(schema.to_string(index=False))
    print(labels.to_string(index=False))
    return summary


def _human(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}K" if n >= 1e3 else str(n)


def _write_latex(schema: pd.DataFrame, labels: pd.DataFrame, summary: Dict, out: Path) -> None:
    """Small LaTeX snippets the milestone report \\input's."""
    lines = [f"\\def\\NumTables{{{summary['tables']}}}",
             f"\\def\\NumRows{{{_human(summary['rows'])}}}",
             f"\\def\\NumCols{{{summary['columns']}}}"]
    for r in summary["labels"]:
        key = ("Outcome" if r["task"] == "study-outcome" else "Adverse") + r["split"].capitalize()
        lines.append(f"\\def\\Rows{key}{{{r['rows']:,}}}".replace(",", "{,}"))
        r = {k: v for k, v in r.items() if not (isinstance(v, float) and v != v)}  # drop NaN
        if "positive_rate" in r:
            lines.append(f"\\def\\Pos{key}{{{100 * r['positive_rate']:.1f}\\%}}")
        if "median" in r:
            lines.append(f"\\def\\Median{key}{{{r['median']:.0f}}}")
            lines.append(f"\\def\\Ptf{key}{{{r['p95']:.0f}}}")
    (out / "stats.tex").write_text("\n".join(lines) + "\n")
