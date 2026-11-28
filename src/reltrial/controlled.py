"""Controlled experiment (A): when does explicit label history help a relational model?

We generate rel-trial-shaped databases where a sponsor's hidden quality ``q`` drives the
success of its trials, with strength ``s`` (logit = s * q + phase effect + noise), in two
regimes:

* ``hidden``: ``q`` is stored nowhere. The only trace of it is the outcomes of the
  sponsor's earlier trials, i.e. labels, which are not in the database.
* ``visible``: ``q`` (plus a little noise) is a column of the sponsors table, two hops from
  the trial (trial <- sponsors_studies -> sponsor): within a 2-layer GNN's reach.

The rule predicts: history helps the GNN in ``hidden`` (growing with s) and not in
``visible``. LightGBM's aggregates never read the sponsor's columns, so for the trees the
history helps in both regimes: "what the model sees" differs by model.

Everything else (sizes, dates, schema, the label task) is identical across regimes.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from relbench.manifest import DatasetManifest, TableSpec, TaskManifest

VAL, TEST = "2016-01-01", "2018-01-01"
MODES = ("hidden", "visible")
STRENGTHS = (0.0, 0.5, 1.0, 2.0)


def name(mode: str, strength: float) -> str:
    return f"controlled-{mode}-s{strength:g}"


def build(root: str | Path, mode: str, strength: float, n_trials: int = 20000,
          n_sponsors: int = 300, seed: int = 0) -> Path:
    """Write one dataset (db parquet files + one task, study-outcome) under ``root``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    rng = np.random.default_rng(seed)  # same seed => same trials, sponsors and noise in every regime
    root = Path(root)
    (root / "db").mkdir(parents=True, exist_ok=True)

    start = pd.to_datetime("2004-01-01") + pd.to_timedelta(rng.integers(0, 365 * 15, n_trials), "D")
    start = np.sort(start.values)
    completion = start + rng.integers(200, 1500, n_trials).astype("timedelta64[D]")
    phases = np.array(["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not Applicable"])
    phase = rng.choice(phases, n_trials)
    studies = pd.DataFrame({
        "nct_id": np.arange(n_trials), "start_date": start, "phase": phase,
        "enrollment": rng.integers(10, 2000, n_trials).astype(float),
        "number_of_arms": rng.integers(1, 5, n_trials).astype(float),
        "has_dmc": rng.choice(["t", "f"], n_trials),
    })

    # Sponsors: skewed sizes (a few big sponsors, many small), latent quality q.
    quality = rng.normal(0, 1.0, n_sponsors)
    size = rng.pareto(1.5, n_sponsors) + 1
    lead = rng.choice(n_sponsors, n_trials, p=size / size.sum())
    sponsors = pd.DataFrame({"sponsor_id": np.arange(n_sponsors),
                             "agency_class": rng.choice(["Industry", "Other", "NIH"], n_sponsors),
                             "founded": rng.integers(1950, 2000, n_sponsors).astype(float)})
    noisy_q = quality + rng.normal(0, 0.3, n_sponsors)  # drawn in both regimes: same rng stream
    if mode == "visible":
        sponsors["quality_score"] = noisy_q
    sponsors_studies = pd.DataFrame({"nct_id": np.arange(n_trials), "sponsor_id": lead,
                                     "lead_or_collaborator": "lead", "date": start})

    n_cond = 40
    conditions = pd.DataFrame({"condition_id": np.arange(n_cond),
                               "category": rng.choice(["oncology", "cardio", "neuro", "other"], n_cond)})
    conditions_studies = pd.DataFrame({"nct_id": np.arange(n_trials),
                                       "condition_id": rng.integers(0, n_cond, n_trials), "date": start})
    eligibilities = pd.DataFrame({"nct_id": np.arange(n_trials),
                                  "gender": rng.choice(["All", "Female", "Male"], n_trials),
                                  "healthy_volunteers": rng.choice(["Yes", "No"], n_trials), "date": start})

    phase_eff = {"Phase 1": -0.3, "Phase 2": 0.0, "Phase 3": 0.4, "Phase 4": 0.2, "Not Applicable": 0.0}
    logit = strength * quality[lead] + pd.Series(phase).map(phase_eff).to_numpy() \
        + rng.normal(0, 0.5, n_trials)
    success = (rng.random(n_trials) < 1 / (1 + np.exp(-logit))).astype(int)
    outcomes = pd.DataFrame({"nct_id": np.arange(n_trials), "outcome_type": "Primary", "date": completion})

    def finish(df, time_col, pkey="id"):
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
        df.insert(0, pkey, np.arange(len(df)))
        return df

    tables = {"studies": studies, "sponsors": sponsors, "conditions": conditions,
              "sponsors_studies": finish(sponsors_studies, "date"),
              "conditions_studies": finish(conditions_studies, "date"),
              "eligibilities": finish(eligibilities, "date"),
              "outcomes": finish(outcomes, "date")}
    for tname, df in tables.items():
        df.to_parquet(root / "db" / f"{tname}.parquet", index=False)
    spec = {
        "studies": TableSpec(pkey="nct_id", time_col="start_date"),
        "sponsors": TableSpec(pkey="sponsor_id"),
        "conditions": TableSpec(pkey="condition_id"),
        "sponsors_studies": TableSpec("id", "date", {"nct_id": "studies", "sponsor_id": "sponsors"}),
        "conditions_studies": TableSpec("id", "date", {"nct_id": "studies", "condition_id": "conditions"}),
        "eligibilities": TableSpec("id", "date", {"nct_id": "studies"}),
        "outcomes": TableSpec("id", "date", {"nct_id": "studies"}),
    }
    DatasetManifest(name=name(mode, strength), val_timestamp=VAL, test_timestamp=TEST,
                    description=f"Controlled experiment: sponsor quality {mode}, strength {strength}.",
                    tables=spec).save(root / "manifest.yaml")

    comp, st = pd.to_datetime(completion), pd.to_datetime(start)
    rows = []
    for t in pd.date_range("2007-01-01", "2019-01-01", freq="YS"):
        m = (comp > t) & (comp <= t + pd.Timedelta(days=365)) & (st <= t)
        rows += [{"timestamp": t, "nct_id": int(i), "outcome": int(success[i])} for i in np.flatnonzero(m)]
    df = pd.DataFrame(rows)
    d = root / "tasks" / "study-outcome"
    d.mkdir(parents=True, exist_ok=True)
    ts = df["timestamp"]
    df[ts < VAL].to_parquet(d / "train.parquet", index=False)
    df[(ts >= VAL) & (ts < TEST)].to_parquet(d / "val.parquet", index=False)
    df[ts >= TEST].to_parquet(d / "test.parquet", index=False)
    TaskManifest(name="study-outcome", kind="external", task_type="binary_classification",
                 entity_table="studies", entity_col="nct_id", target_col="outcome",
                 time_col="timestamp", timedelta="365 days").save(d / "manifest.yaml")
    return root


def build_all(root: str | Path, modes: Sequence[str] = MODES,
              strengths: Sequence[float] = STRENGTHS, **kw) -> List[Path]:
    return [build(Path(root) / name(m, s), m, s, **kw) for m, s in itertools.product(modes, strengths)]


PAIRS = {"GNN": ("rdl_sage_notext_hist", "rdl_sage_notext"),
         "GNN, history in graph": ("rdl_sage_notext_ghist", "rdl_sage_notext"),
         "LightGBM": ("lgbm_agg", "lgbm_agg_nohist")}


def summarize(data_root: str | Path, results_root: str | Path, out_dir: str | Path,
              split: str = "test", n_boot: int = 1000,
              modes: Iterable[str] = MODES, strengths: Iterable[float] = STRENGTHS) -> pd.DataFrame:
    """History gain (with - without, paired bootstrap CI) per regime, strength and model."""
    from . import data as D
    from .evaluate import compute_metrics
    from .significance import compare, load_seed_preds

    rows = []
    for mode, s in itertools.product(modes, strengths):
        ds = Path(data_root) / name(mode, s)
        preds = Path(results_root) / name(mode, s) / "preds"
        if not preds.exists():
            continue
        b = D.load_bundle("study-outcome", str(ds))
        y = b.split(split).df[b.task.target_col].to_numpy(dtype=float)
        for family, (a, base) in PAIRS.items():
            A = load_seed_preds(preds, "study-outcome", a, split)
            B = load_seed_preds(preds, "study-outcome", base, split)
            if A is None or B is None:
                continue
            r = compare(y, A, B, "roc_auc", n_boot=n_boot)
            auc_a = float(np.mean([compute_metrics(b.task.task_type.value, y, p)["roc_auc"] for p in A]))
            auc_b = float(np.mean([compute_metrics(b.task.task_type.value, y, p)["roc_auc"] for p in B]))
            rows.append({"mode": mode, "strength": s, "model": family, "auc_with": auc_a,
                         "auc_without": auc_b, **r})
    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "controlled.csv", index=False)
    if not df.empty:
        md = ["| Sponsor quality | Strength | Model | AUROC without history | with history | Gain | 95% CI |",
              "|---|---|---|---|---|---|---|"]
        for r in df.sort_values(["mode", "model", "strength"]).itertuples():
            md.append(f"| {r.mode} | {r.strength:g} | {r.model} | {r.auc_without:.3f} | {r.auc_with:.3f} | "
                      f"{r.diff:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] |")
        md += ["", f"Test AUROC, mean of seeds; gain = with - without history, paired bootstrap "
               f"({n_boot} resamples) on the {split} split."]
        (out / "controlled.md").write_text("\n".join(md) + "\n")
    return df


def plot(df: pd.DataFrame, out: str | Path) -> None:
    """Two panels (hidden | visible): history gain vs. signal strength, one line per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figures import FAMILY, GRID, INK, INK2

    # accent blue = GNN (history at the head), lighter blue = GNN (history in the graph), grey = trees
    colors = {"GNN": FAMILY["LightGBM"], "GNN, history in graph": "#8fbbef", "LightGBM": "#9c9a8e"}
    titles = {"hidden": "Quality stored nowhere (only in past labels)",
              "visible": "Quality stored on the sponsor (2 hops away)"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    lo = min(0.0, df["ci_low"].min()) - 0.005
    hi = df["ci_high"].max() + 0.01
    for ax, mode in zip(axes, MODES):
        d = df[df["mode"] == mode]
        ax.axhline(0, color="#bdbdbd", lw=1)
        for fam in ("GNN", "GNN, history in graph", "LightGBM"):
            g = d[d["model"] == fam].sort_values("strength")
            if g.empty:
                continue
            ax.errorbar(g["strength"], g["diff"], yerr=[g["diff"] - g["ci_low"], g["ci_high"] - g["diff"]],
                        color=colors[fam], lw=2, marker="o", ms=6, capsize=3, label=fam)
            last = g.iloc[-1]
            ax.annotate(fam, (last["strength"], last["diff"]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8, color=colors[fam])
        ax.set_title(titles[mode], loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("How strongly sponsor quality drives the outcome", fontsize=8, color=INK2)
        ax.set_ylim(lo, hi)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=7)
        ax.set_xlim(-0.15, max(STRENGTHS) + 0.45)
    axes[0].set_ylabel("Test AUROC gain from label history", fontsize=8, color=INK2)
    fig.suptitle("Gain from label history as the sponsor signal grows",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

