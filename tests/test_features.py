"""Leakage tests for the hand-engineered features: nothing from after the seed time."""

import numpy as np
import pandas as pd
import pytest

from relbench.base import Database, Table
from reltrial.features import FeatureBuilder, count_before, sum_before


def test_count_before_strict_and_inclusive():
    keys = np.array([1, 1, 1, 2])
    days = np.array([10, 20, 30, 15])
    q_keys = np.array([1, 1, 2, 3])
    q_days = np.array([20, 20, 15, 99])
    assert count_before(keys, days, q_keys, q_days, inclusive=False).tolist() == [1, 1, 0, 0]
    assert count_before(keys, days, q_keys, q_days, inclusive=True).tolist() == [2, 2, 1, 0]


def test_sum_before():
    n, s = sum_before(np.array([5, 5, 5]), np.array([1, 2, 3]), np.array([1.0, 0.0, 1.0]),
                      np.array([5, 5]), np.array([2, 3]), inclusive=True)
    assert n.tolist() == [2, 3]
    assert s.tolist() == [1.0, 2.0]


def _tiny_db():
    d = pd.to_datetime
    studies = pd.DataFrame({"nct_id": [0, 1, 2], "start_date": d(["2010-01-01", "2012-01-01", "2014-01-01"]),
                            "phase": ["Phase 2", "Phase 3", "Phase 2"], "enrollment": [10.0, 20.0, 30.0]})
    sponsors = pd.DataFrame({"sponsor_id": [0], "name": ["Acme"]})
    links = pd.DataFrame({"id": [0, 1, 2], "nct_id": [0, 1, 2], "sponsor_id": [0, 0, 0],
                          "date": d(["2010-01-01", "2012-01-01", "2014-01-01"])})
    outcomes = pd.DataFrame({"id": [0, 1], "nct_id": [0, 1], "date": d(["2011-06-01", "2013-06-01"])})
    return Database({
        "studies": Table(studies, {}, "nct_id", "start_date"),
        "sponsors": Table(sponsors, {}, "sponsor_id", None),
        "sponsors_studies": Table(links, {"nct_id": "studies", "sponsor_id": "sponsors"}, "id", "date"),
        "outcomes": Table(outcomes, {"nct_id": "studies"}, "id", "date"),
    })


@pytest.fixture
def builder():
    history = pd.DataFrame({"timestamp": pd.to_datetime(["2010-06-01", "2012-06-01"]),
                            "nct_id": [0, 1], "outcome": [1.0, 0.0]})
    fb = FeatureBuilder(_tiny_db(), "studies", "nct_id", "timestamp", "outcome",
                        pd.Timedelta(days=365), feature_set="agg", label_history=history,
                        text_dims=0, smoothing=0.0)
    return fb.fit(pd.DataFrame({"timestamp": pd.to_datetime(["2015-01-01"]), "nct_id": [2]}))


def test_labels_are_used_only_once_known(builder):
    # Trial 0's label (seed 2010-06-01) becomes known 2011-06-01; trial 1's on 2013-06-01.
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2011-01-01", "2012-01-01", "2014-01-01"]),
                         "nct_id": [2, 2, 2]})
    rows.loc[:, "nct_id"] = [2, 2, 2]
    f = builder.transform(rows)
    # Trial 2 starts 2014-01-01, so its sponsor link is invisible before then.
    assert f["n_sponsor"].tolist() == [0, 0, 1]
    # At 2014-01-01 both earlier labels are known: success rate 1/2 (smoothing off).
    assert f["sponsor_hist_target_mean"].iloc[2] == pytest.approx(0.5)
    assert f["sponsor_hist_n_sum"].iloc[2] == 2


def test_history_excludes_labels_not_yet_known(builder):
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2014-01-01"]), "nct_id": [2]})
    hist = builder.label_history.copy()
    hist.loc[len(hist)] = [pd.Timestamp("2013-12-01"), 1, 1.0]  # known only on 2014-12-01
    builder.label_history = hist
    f = builder.transform(rows)
    assert f["sponsor_hist_n_sum"].iloc[0] == 2  # the late label is ignored


def test_satellite_rows_after_seed_time_are_not_counted(builder):
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2011-01-01", "2012-01-01"]), "nct_id": [0, 0]})
    f = builder.transform(rows)
    # Trial 0's outcome row is dated 2011-06-01: invisible at 2011-01-01, visible at 2012.
    assert f["n_outcomes"].tolist() == [0, 1]


def test_prior_trials_counts_only_earlier_starts(builder):
    rows = pd.DataFrame({"timestamp": pd.to_datetime(["2014-01-01", "2015-01-01"]), "nct_id": [2, 2]})
    f = builder.transform(rows)
    # Before 2014-01-01 the sponsor started trials 0 and 1; by 2015 also trial 2.
    assert f["sponsor_prior_trials_mean"].tolist() == [2, 3]
