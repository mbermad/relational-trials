"""Leakage-safe tabular features for the LightGBM baselines.

Two feature sets, matching the proposal:

* ``table``: the target trial's own row in ``studies`` (categoricals, numbers, dates) plus
  a compact TF-IDF/SVD summary of its title text.
* ``agg``: ``table`` + hand-engineered joins over the database (Deep Feature Synthesis
  style): one-to-one satellite tables (``designs``, ``eligibilities``), counts of linked
  rows, and the historical record of each linked sponsor / condition / facility /
  intervention (how many trials it ran before, and how those trials turned out).

Every feature for a row with seed time ``t`` uses only information available at ``t``:

* database rows count only if their own timestamp is ``<= t`` (the same rule the GNN's
  temporal neighbor sampler applies);
* a label of another trial counts only once it is *known*: its label time plus the task's
  ``timedelta`` must be ``<= t``. This is the time-censoring RelBench's own examples use
  when they feed labels back in as features.

``tests/test_features.py`` checks both rules on hand-built examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from relbench.base import Database

KEY_SCALE = 10**6  # days since epoch < 10**6 for any realistic date


def _days(ts: pd.Series) -> np.ndarray:
    """Timestamps -> integer days since the epoch (NaT -> very large)."""
    ts = pd.Series(pd.to_datetime(ts))
    # Unit-agnostic (pandas 3 parses to microseconds, pandas 2 to nanoseconds).
    days = (ts - pd.Timestamp("1970-01-01")) // pd.Timedelta(days=1)
    out = np.where(ts.isna().to_numpy(), KEY_SCALE - 1, days.fillna(0).to_numpy())
    return out.astype(np.int64)


# --------------------------------------------------------------------------------------
# Column typing
# --------------------------------------------------------------------------------------
@dataclass
class ColumnKinds:
    numeric: List[str] = field(default_factory=list)
    categorical: List[str] = field(default_factory=list)
    datetime: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)


def infer_kinds(df: pd.DataFrame, exclude: set, max_categories: int = 100) -> ColumnKinds:
    kinds = ColumnKinds()
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            kinds.categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            kinds.numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            kinds.datetime.append(col)
        else:
            nunique = s.nunique(dropna=True)
            if nunique == 0:
                continue
            if nunique <= max_categories:
                kinds.categorical.append(col)
            else:
                kinds.text.append(col)
    return kinds


def _table_meta(db: Database, name: str):
    t = db.table_dict[name]
    return t.df, t.pkey_col, t.time_col, dict(t.fkey_col_to_pkey_table)


# --------------------------------------------------------------------------------------
# Historical aggregates over junction tables (vectorized)
# --------------------------------------------------------------------------------------
def count_before(keys: np.ndarray, days: np.ndarray, q_keys: np.ndarray,
                 q_days: np.ndarray, inclusive: bool) -> np.ndarray:
    """For each query (key, day): number of events of that key with day < q (or <= q).

    Events are encoded as ``key * KEY_SCALE + day`` in one sorted array, so a query is two
    binary searches. Works for millions of events in well under a second.
    """
    enc = np.sort(keys.astype(np.int64) * KEY_SCALE + days)
    side = "right" if inclusive else "left"
    hi = np.searchsorted(enc, q_keys.astype(np.int64) * KEY_SCALE + q_days, side=side)
    lo = np.searchsorted(enc, q_keys.astype(np.int64) * KEY_SCALE, side="left")
    return hi - lo


def sum_before(keys: np.ndarray, days: np.ndarray, values: np.ndarray,
               q_keys: np.ndarray, q_days: np.ndarray, inclusive: bool):
    """Like ``count_before`` but also returns the sum of ``values`` over those events."""
    enc = keys.astype(np.int64) * KEY_SCALE + days
    order = np.argsort(enc, kind="stable")
    enc, vals = enc[order], np.asarray(values, dtype=float)[order]
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    side = "right" if inclusive else "left"
    hi = np.searchsorted(enc, q_keys.astype(np.int64) * KEY_SCALE + q_days, side=side)
    lo = np.searchsorted(enc, q_keys.astype(np.int64) * KEY_SCALE, side="left")
    return hi - lo, csum[hi] - csum[lo]


@dataclass
class Junction:
    table: str
    entity_fk: str  # column pointing at the entity table (e.g. nct_id)
    other_fk: str  # column pointing at the other entity (e.g. sponsor_id)
    time_col: Optional[str]


def find_junctions(db: Database, entity_table: str) -> List[Junction]:
    out = []
    for name, t in db.table_dict.items():
        fks = dict(t.fkey_col_to_pkey_table)
        to_entity = [c for c, tgt in fks.items() if tgt == entity_table]
        others = [c for c, tgt in fks.items() if tgt != entity_table]
        if len(to_entity) == 1 and len(others) == 1:
            other_table = db.table_dict[fks[others[0]]]
            # Skip "junctions" whose other side hangs off the entity itself
            # (e.g. outcome_analyses -> outcomes -> studies): no shared entity there.
            if entity_table in dict(other_table.fkey_col_to_pkey_table).values():
                continue
            out.append(Junction(name, to_entity[0], others[0], t.time_col))
    return out


def find_satellites(db: Database, entity_table: str) -> List[str]:
    """Tables whose only foreign key points at the entity table (designs, outcomes, ...)."""
    out = []
    for name, t in db.table_dict.items():
        fks = dict(t.fkey_col_to_pkey_table)
        if name != entity_table and fks and set(fks.values()) == {entity_table}:
            if len(fks) == 1:
                out.append(name)
    return out


# --------------------------------------------------------------------------------------
# The feature builder
# --------------------------------------------------------------------------------------
class FeatureBuilder:
    """Fit on the training rows, then transform any split into a feature frame."""

    def __init__(self, db: Database, entity_table: str, entity_col: str, time_col: str,
                 target_col: str, timedelta: pd.Timedelta, feature_set: str = "table",
                 label_history: Optional[pd.DataFrame] = None, text_dims: int = 32,
                 smoothing: float = 5.0):
        if feature_set not in ("table", "agg"):
            raise ValueError("feature_set must be 'table' or 'agg'")
        self.db = db
        self.entity_table, self.entity_col = entity_table, entity_col
        self.time_col, self.target_col = time_col, target_col
        self.timedelta = pd.Timedelta(timedelta)
        self.feature_set = feature_set
        self.label_history = label_history
        self.text_dims = text_dims
        self.smoothing = smoothing
        self._fitted = False

    # ---- fitting ----------------------------------------------------------------------
    def fit(self, train_rows: pd.DataFrame) -> "FeatureBuilder":
        edf, pkey, etime, efks = _table_meta(self.db, self.entity_table)
        exclude = {pkey, *efks.keys()}
        self.entity_kinds = infer_kinds(edf, exclude)
        self.entity_time = etime

        base = self._entity_frame(train_rows)
        self.categories: Dict[str, pd.Index] = {
            c: pd.Index(base[c].astype("string").dropna().unique())
            for c in self.entity_kinds.categorical
        }
        self._fit_text(base)

        if self.feature_set == "agg":
            self.junctions = find_junctions(self.db, self.entity_table)
            self.satellites = find_satellites(self.db, self.entity_table)
            self.one_to_one = {}
            for name in self.satellites:
                df, spk, stime, sfks = _table_meta(self.db, name)
                fk = next(iter(sfks))
                per_entity = df.groupby(fk).size()
                if len(per_entity) and per_entity.mean() <= 1.05:
                    kinds = infer_kinds(df, {spk, fk, stime})
                    self.one_to_one[name] = (fk, stime, kinds)
                    for c in kinds.categorical:
                        self.categories[f"{name}.{c}"] = pd.Index(
                            df[c].astype("string").dropna().unique())
            if self.label_history is not None:
                self.prior = float(self.label_history[self.target_col].mean())
        self._fitted = True
        return self

    def _fit_text(self, base: pd.DataFrame) -> None:
        self.text_cols = [c for c in self.entity_kinds.text
                          if c in ("brief_title", "official_title")] or self.entity_kinds.text[:2]
        self.tfidf = self.svd = None
        if not self.text_cols or self.text_dims <= 0:
            return
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = self._docs(base)
        self.tfidf = TfidfVectorizer(max_features=20_000, min_df=2, ngram_range=(1, 2),
                                     stop_words="english", sublinear_tf=True)
        try:
            X = self.tfidf.fit_transform(docs)
        except ValueError:  # empty vocabulary (tiny synthetic data)
            self.tfidf = None
            return
        k = min(self.text_dims, max(1, X.shape[1] - 1), max(1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(X)

    def _docs(self, frame: pd.DataFrame) -> List[str]:
        return frame[self.text_cols].astype("string").fillna("").agg(" ".join, axis=1).tolist()

    # ---- transforming -----------------------------------------------------------------
    def _entity_frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        edf, pkey, _, _ = _table_meta(self.db, self.entity_table)
        left = rows[[self.entity_col, self.time_col]].reset_index(drop=True)
        merged = left.merge(edf, how="left", left_on=self.entity_col, right_on=pkey,
                            suffixes=("", "__entity"))
        assert len(merged) == len(left), "entity join changed the number of rows"
        return merged

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted, "call fit() first"
        base = self._entity_frame(rows)
        t = pd.to_datetime(base[self.time_col])
        feats = pd.DataFrame(index=base.index)

        k = self.entity_kinds
        for c in k.numeric:
            feats[c] = pd.to_numeric(base[c], errors="coerce").astype(float)
        for c in k.categorical:
            feats[c] = pd.Categorical(base[c].astype("string"), categories=self.categories[c])
        for c in k.datetime:
            d = pd.to_datetime(base[c])
            feats[f"{c}_year"] = d.dt.year.astype(float)
            feats[f"days_{c}_to_seed"] = (t - d).dt.days.astype(float)
        feats["seed_year"] = t.dt.year.astype(float)
        if self.tfidf is not None and self.svd is not None:
            Z = self.svd.transform(self.tfidf.transform(self._docs(base)))
            for i in range(Z.shape[1]):
                feats[f"text_svd_{i}"] = Z[:, i]

        if self.feature_set == "agg":
            feats = pd.concat([feats, self._agg_features(base, t)], axis=1)
        return feats

    def _agg_features(self, base: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame(index=base.index)
        q_entity = base[self.entity_col].to_numpy()
        q_day = _days(t)
        row_ids = np.arange(len(base))

        # One-to-one satellites (designs, eligibilities): copy their columns if the row
        # existed at the seed time.
        for name, (fk, stime, kinds) in self.one_to_one.items():
            df = self.db.table_dict[name].df.drop_duplicates(fk)
            m = base[[self.entity_col]].merge(df, how="left", left_on=self.entity_col,
                                              right_on=fk)
            visible = np.ones(len(m), dtype=bool)
            if stime is not None:
                visible = (_days(m[stime]) <= q_day)
            for c in kinds.numeric:
                out[f"{name}.{c}"] = np.where(visible, pd.to_numeric(m[c], errors="coerce"), np.nan)
            for c in kinds.categorical:
                vals = m[c].astype("string").where(visible)
                out[f"{name}.{c}"] = pd.Categorical(vals, categories=self.categories[f"{name}.{c}"])

        # Count rows in every satellite (outcomes, drop_withdrawals, ...) visible at t.
        for name in self.satellites:
            df, _, stime, sfks = _table_meta(self.db, name)
            fk = next(iter(sfks))
            if stime is None:
                continue
            out[f"n_{name}"] = count_before(df[fk].to_numpy(), _days(df[stime]),
                                            q_entity, q_day, inclusive=True)

        hist = self._label_events()
        for j in self.junctions:
            jdf = self.db.table_dict[j.table].df
            jdays = _days(jdf[j.time_col]) if j.time_col else np.zeros(len(jdf), np.int64)
            # Links of each query trial that exist at t: (row_id, other_id).
            links = pd.DataFrame({self.entity_col: jdf[j.entity_fk].to_numpy(),
                                  "other": jdf[j.other_fk].to_numpy(), "ldays": jdays})
            q = pd.DataFrame({"row": row_ids, self.entity_col: q_entity, "qday": q_day})
            pairs = q.merge(links, on=self.entity_col, how="inner")
            pairs = pairs[(pairs["ldays"] <= pairs["qday"]) & pairs["other"].notna()]
            other = pairs["other"].to_numpy().astype(np.int64)
            short = j.other_fk.replace("_id", "")

            # How many trials the linked entity was part of before t.
            prior = count_before(links["other"].fillna(-1).to_numpy().astype(np.int64),
                                 jdays, other, pairs["qday"].to_numpy(), inclusive=False)
            pairs = pairs.assign(prior=prior)

            # The linked entity's track record: labels of its trials known by t.
            if hist is not None:
                hl = links.merge(hist, on=self.entity_col, how="inner")
                hl = hl[hl["other"].notna()]
                n, s = sum_before(hl["other"].to_numpy().astype(np.int64),
                                  hl["avail"].to_numpy(), hl["y"].to_numpy(), other,
                                  pairs["qday"].to_numpy(), inclusive=True)
                m = self.smoothing
                with np.errstate(invalid="ignore", divide="ignore"):
                    rate = (s + m * self.prior) / (n + m)
                pairs = pairs.assign(hist_n=n, hist_rate=rate)

            g = pairs.groupby("row")
            out[f"n_{short}"] = g.size().reindex(row_ids).fillna(0).to_numpy()
            out[f"{short}_prior_trials_mean"] = g["prior"].mean().reindex(row_ids).to_numpy()
            out[f"{short}_prior_trials_max"] = g["prior"].max().reindex(row_ids).to_numpy()
            if hist is not None:
                out[f"{short}_hist_target_mean"] = g["hist_rate"].mean().reindex(row_ids).to_numpy()
                out[f"{short}_hist_target_max"] = g["hist_rate"].max().reindex(row_ids).to_numpy()
                out[f"{short}_hist_n_sum"] = g["hist_n"].sum().reindex(row_ids).to_numpy()
        return out

    def _label_events(self) -> Optional[pd.DataFrame]:
        """Known labels as events: (entity, day the label becomes known, value)."""
        if self.label_history is None:
            return None
        h = self.label_history
        avail = pd.to_datetime(h[self.time_col]) + self.timedelta
        return pd.DataFrame({self.entity_col: h[self.entity_col].to_numpy(),
                             "avail": _days(avail), "y": h[self.target_col].astype(float).to_numpy()})
