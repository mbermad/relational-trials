"""A tiny synthetic dataset with rel-trial's layout, for offline end-to-end tests.

It mirrors the real schema's shape (studies at the center; one-to-one satellites dated at
trial start; outcomes dated at completion; junction tables to sponsors, conditions and
facilities) and plants a learnable signal: a sponsor's latent quality drives its trials'
outcomes, so relational features and GNNs should beat chance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from relbench.manifest import DatasetManifest, TableSpec, TaskManifest

VAL, TEST = "2018-01-01", "2019-01-01"


def build(root: str | Path, n_trials: int = 1500, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "db").mkdir(parents=True, exist_ok=True)

    # ---- studies (entity table), time-sorted so pkeys are 0..n-1 in time order -------
    start = pd.to_datetime("2005-01-01") + pd.to_timedelta(rng.integers(0, 365 * 14, n_trials), "D")
    start = np.sort(start.values)
    duration = rng.integers(200, 1500, n_trials)
    completion = start + duration.astype("timedelta64[D]")
    phases = np.array(["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not Applicable"])
    words = np.array(["cancer", "diabetes", "heart", "pain", "vaccine", "safety", "dose",
                      "randomized", "placebo", "efficacy", "pediatric", "adult"])
    titles = [" ".join(rng.choice(words, 5)) for _ in range(n_trials)]
    studies = pd.DataFrame({
        "nct_id": np.arange(n_trials),
        "start_date": start,
        "phase": rng.choice(phases, n_trials),
        "enrollment": rng.integers(10, 2000, n_trials).astype(float),
        "number_of_arms": rng.integers(1, 5, n_trials).astype(float),
        "has_dmc": rng.choice(["t", "f"], n_trials),
        "brief_title": titles,
        "official_title": [t + " study" for t in titles],
    })

    # ---- sponsors with latent quality; each trial has one lead sponsor -----------------
    n_sp = 60
    quality = rng.normal(0, 1.2, n_sp)
    sponsors = pd.DataFrame({"sponsor_id": np.arange(n_sp),
                             "name": [f"Sponsor {i}" for i in range(n_sp)],
                             "agency_class": rng.choice(["Industry", "Other", "NIH"], n_sp)})
    lead = rng.integers(0, n_sp, n_trials)
    sponsors_studies = pd.DataFrame({"nct_id": np.arange(n_trials), "sponsor_id": lead,
                                     "lead_or_collaborator": "lead", "date": start})

    n_cond = 25
    conditions = pd.DataFrame({"condition_id": np.arange(n_cond),
                               "mesh_term": [f"Condition {i}" for i in range(n_cond)]})
    cond = rng.integers(0, n_cond, n_trials)
    conditions_studies = pd.DataFrame({"nct_id": np.arange(n_trials), "condition_id": cond, "date": start})

    n_fac = 80
    facilities = pd.DataFrame({"facility_id": np.arange(n_fac), "name": [f"Site {i}" for i in range(n_fac)],
                               "country": rng.choice(["US", "FR", "DE", "CN"], n_fac)})
    fac_rows = [(i, f, start[i]) for i in range(n_trials) for f in rng.choice(n_fac, rng.integers(1, 4), replace=False)]
    facilities_studies = pd.DataFrame(fac_rows, columns=["nct_id", "facility_id", "date"])

    designs = pd.DataFrame({"nct_id": np.arange(n_trials),
                            "allocation": rng.choice(["Randomized", "Non-Randomized"], n_trials),
                            "masking": rng.choice(["None", "Double", "Quadruple"], n_trials), "date": start})
    eligibilities = pd.DataFrame({"nct_id": np.arange(n_trials), "gender": rng.choice(["All", "Female", "Male"], n_trials),
                                  "healthy_volunteers": rng.choice(["Yes", "No"], n_trials), "date": start})

    # ---- labels driven by sponsor quality + phase -------------------------------------
    phase_eff = {"Phase 1": -0.3, "Phase 2": 0.0, "Phase 3": 0.4, "Phase 4": 0.2, "Not Applicable": 0.0}
    logit = quality[lead] + studies["phase"].map(phase_eff).to_numpy() + rng.normal(0, 0.5, n_trials)
    success = (rng.random(n_trials) < 1 / (1 + np.exp(-logit))).astype(int)
    adverse = rng.poisson(np.exp(1.5 + 0.6 * quality[lead] + 0.0005 * studies["enrollment"].to_numpy()))
    outcomes = pd.DataFrame({"nct_id": np.arange(n_trials), "outcome_type": "Primary",
                             "title": "primary endpoint", "date": completion})

    # Contract: tables with a time column sorted by it, pkeys 0..n-1 in row order.
    def finish(df, time_col, pkey="id"):
        df = df.sort_values(time_col, kind="stable").reset_index(drop=True)
        df.insert(0, pkey, np.arange(len(df)))
        return df

    tables = {
        "studies": studies,
        "sponsors": sponsors, "conditions": conditions, "facilities": facilities,
        "sponsors_studies": finish(sponsors_studies, "date"),
        "conditions_studies": finish(conditions_studies, "date"),
        "facilities_studies": finish(facilities_studies, "date"),
        "designs": finish(designs, "date"),
        "eligibilities": finish(eligibilities, "date"),
        "outcomes": finish(outcomes, "date"),
    }
    for name, df in tables.items():
        df.to_parquet(root / "db" / f"{name}.parquet", index=False)

    spec = {
        "studies": TableSpec(pkey="nct_id", time_col="start_date"),
        "sponsors": TableSpec(pkey="sponsor_id"),
        "conditions": TableSpec(pkey="condition_id"),
        "facilities": TableSpec(pkey="facility_id"),
        "sponsors_studies": TableSpec("id", "date", {"nct_id": "studies", "sponsor_id": "sponsors"}),
        "conditions_studies": TableSpec("id", "date", {"nct_id": "studies", "condition_id": "conditions"}),
        "facilities_studies": TableSpec("id", "date", {"nct_id": "studies", "facility_id": "facilities"}),
        "designs": TableSpec("id", "date", {"nct_id": "studies"}),
        "eligibilities": TableSpec("id", "date", {"nct_id": "studies"}),
        "outcomes": TableSpec("id", "date", {"nct_id": "studies"}),
    }
    DatasetManifest(name="rel-trial-synthetic", val_timestamp=VAL, test_timestamp=TEST,
                    description="Synthetic rel-trial-shaped data for tests.",
                    tables=spec).save(root / "manifest.yaml")

    # ---- tasks: yearly seed times; entity = trials completing in (t, t + 1 year] ------
    comp = pd.to_datetime(completion)
    seeds = pd.date_range("2008-01-01", "2020-01-01", freq="YS")

    def labels(target):
        rows = []
        for t in seeds:
            m = (comp > t) & (comp <= t + pd.Timedelta(days=365)) & (pd.to_datetime(start) <= t)
            for i in np.flatnonzero(m):
                rows.append({"timestamp": t, "nct_id": int(i), target[0]: target[1][i]})
        return pd.DataFrame(rows)

    for name, task_type, target in [
        ("study-outcome", "binary_classification", ("outcome", success)),
        ("study-adverse", "regression", ("num_of_adverse_events", adverse.astype(float))),
    ]:
        d = root / "tasks" / name
        d.mkdir(parents=True, exist_ok=True)
        df = labels(target)
        ts = df["timestamp"]
        df[ts < VAL].to_parquet(d / "train.parquet", index=False)
        df[(ts >= VAL) & (ts < TEST)].to_parquet(d / "val.parquet", index=False)
        df[ts >= TEST].to_parquet(d / "test.parquet", index=False)
        TaskManifest(name=name, kind="external", task_type=task_type, entity_table="studies",
                     entity_col="nct_id", target_col=target[0], time_col="timestamp",
                     timedelta="365 days").save(d / "manifest.yaml")
    return root


if __name__ == "__main__":
    import sys

    print(build(sys.argv[1] if len(sys.argv) > 1 else "data/rel-trial-synthetic"))
