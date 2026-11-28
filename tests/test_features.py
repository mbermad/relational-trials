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


def test_own_history_uses_only_known_labels():
    d = pd.to_datetime
    drivers = pd.DataFrame({"driverId": [0, 1], "nationality": ["A", "B"]})
    races = pd.DataFrame({"raceId": [0], "date": d(["2010-01-01"])})
    results = pd.DataFrame({"resultId": [0], "raceId": [0], "driverId": [0], "date": d(["2010-01-01"])})
    db = Database({"drivers": Table(drivers, {}, "driverId", None),
                   "races": Table(races, {}, "raceId", "date"),
                   "results": Table(results, {"raceId": "races", "driverId": "drivers"}, "id", "date")})
    # Driver 0 labelled at three seed times; each label is known 30 days later.
    hist = pd.DataFrame({"timestamp": d(["2010-01-01", "2010-02-01", "2010-03-01"]),
                         "driverId": [0, 0, 0], "dnf": [1.0, 0.0, 1.0]})
    fb = FeatureBuilder(db, "drivers", "driverId", "timestamp", "dnf", pd.Timedelta(days=30),
                        feature_set="agg", label_history=hist, text_dims=0, smoothing=0.0)
    fb.fit(hist)
    q = pd.DataFrame({"timestamp": d(["2010-01-15", "2010-02-20", "2010-03-20"]), "driverId": [0, 0, 0]})
    f = fb.transform(q)
    assert f["own_hist_n_sum"].tolist() == [0, 1, 2]
    assert np.isnan(f["own_hist_target_last"].iloc[0])
    assert f["own_hist_target_last"].iloc[1] == 1.0 and f["own_hist_target_last"].iloc[2] == 0.0
    assert f["own_hist_target_mean"].iloc[2] == pytest.approx(0.5)


def test_multi_key_tables_yield_one_junction_per_key():
    from reltrial.features import find_junctions

    d = pd.to_datetime
    db = Database({
        "drivers": Table(pd.DataFrame({"driverId": [0]}), {}, "driverId", None),
        "races": Table(pd.DataFrame({"raceId": [0], "date": d(["2010-01-01"])}), {}, "raceId", "date"),
        "constructors": Table(pd.DataFrame({"constructorId": [0]}), {}, "constructorId", None),
        "results": Table(pd.DataFrame({"resultId": [0], "raceId": [0], "driverId": [0], "constructorId": [0],
                                       "date": d(["2010-01-01"])}),
                         {"raceId": "races", "driverId": "drivers", "constructorId": "constructors"}, "resultId", "date"),
        "standings": Table(pd.DataFrame({"sId": [0], "raceId": [0], "driverId": [0], "date": d(["2010-01-01"])}),
                           {"raceId": "races", "driverId": "drivers"}, "sId", "date"),
    })
    shorts = sorted(j.short for j in find_junctions(db, "drivers"))
    assert shorts == ["constructor", "results_race", "standings_race"]


def test_counts_are_right_for_dates_before_1970():
    """rel-f1 starts in 1950: days must stay non-negative or keys bleed into each other."""
    from reltrial.features import _days

    ev = pd.to_datetime(pd.Series(["1950-05-13", "1951-01-01", "1960-01-01", "1955-01-01"]))
    keys = np.array([1, 1, 1, 2])
    q = pd.to_datetime(pd.Series(["1950-01-01", "1952-01-01", "1961-01-01", "1956-01-01"]))
    qk = np.array([1, 1, 1, 2])
    n = count_before(keys, _days(ev), qk, _days(q), inclusive=True)
    assert n.tolist() == [0, 2, 3, 1]
    cnt, s = sum_before(keys, _days(ev), np.array([1.0, 0.0, 1.0, 1.0]), qk, _days(q), inclusive=True)
    assert cnt.tolist() == [0, 2, 3, 1] and s.tolist() == [0.0, 1.0, 2.0, 1.0]


def test_activity_counts_for_stack_like_tables():
    """Posts / comments point at users AND at posts (which point back at users): no junction,
    so they get activity counts; only rows before the seed time count."""
    from reltrial.features import find_activity_tables

    d = pd.to_datetime
    users = pd.DataFrame({"Id": [0, 1], "Location": ["x", "y"]})
    posts = pd.DataFrame({"PostId": [0, 1, 2], "OwnerUserId": [0, 0, np.nan],
                          "ParentId": [np.nan, 0, 0],
                          "CreationDate": d(["2020-01-01", "2020-03-01", "2020-03-05"])})
    comments = pd.DataFrame({"CId": [0, 1, 2], "UserId": [1, 1, 0], "PostId": [0, 0, 1],
                             "CreationDate": d(["2020-01-10", "2020-03-20", "2020-03-02"])})
    db = Database({"users": Table(users, {}, "Id", None),
                   "posts": Table(posts, {"OwnerUserId": "users", "ParentId": "posts"},
                                  "PostId", "CreationDate"),
                   "comments": Table(comments, {"UserId": "users", "PostId": "posts"},
                                     "CId", "CreationDate")})
    assert find_activity_tables(db, "users") == [("comments", "UserId"), ("posts", "OwnerUserId")]
    rows = pd.DataFrame({"timestamp": d(["2020-04-01", "2020-04-01", "2020-01-15"]),
                         "Id": [0, 1, 1], "y": [1.0, 0.0, 1.0]})
    fb = FeatureBuilder(db, "users", "Id", "timestamp", "y", pd.Timedelta(days=40),
                        feature_set="agg", text_dims=0, activity=True).fit(rows)
    f = fb.transform(rows)
    assert f["n_posts_OwnerUser"].tolist() == [2, 0, 0]
    assert f["n_posts_OwnerUser_last_window"].tolist() == [1, 0, 0]  # window = (t - 40 days, t]
    assert f["n_comments_User"].tolist() == [1, 2, 1]
    assert f["n_comments_User_last_window"].tolist() == [1, 1, 1]
    assert f["days_since_comments_User"].tolist()[1] == 12  # 2020-03-20 -> 2020-04-01
    off = FeatureBuilder(db, "users", "Id", "timestamp", "y", pd.Timedelta(days=31),
                         feature_set="agg", text_dims=0).fit(rows).transform(rows)
    assert not any(c.startswith("n_posts") for c in off.columns)  # default: unchanged features


def test_latest_rows_keeps_whole_recent_seed_times():
    from reltrial.data import latest_rows

    d = pd.to_datetime
    df = pd.DataFrame({"timestamp": d(["2020-01-01"] * 3 + ["2020-04-01"] * 3 + ["2020-07-01"] * 3),
                       "Id": list(range(9)), "y": [0.0] * 9})
    sub = latest_rows(Table(df, {"Id": "users"}, None, "timestamp"), 4).df
    assert sorted(sub["timestamp"].unique()) == list(d(["2020-04-01", "2020-07-01"]))


def test_junction_history_columns_only_use_labels_known_by_the_row_date(tmp_path):
    """B: history written onto junction rows must match a brute-force as-of computation."""
    import synthetic

    from reltrial import data
    from reltrial.features import junction_history_columns

    b = data.load_bundle("study-outcome", str(synthetic.build(tmp_path / "rel-trial-synthetic")))
    t, db = b.task, b.dataset.get_db()
    prior = float(b.train.df[t.target_col].mean())
    extra = junction_history_columns(db, t.entity_table, t.entity_col, b.history_rows, t.time_col,
                                     t.target_col, t.timedelta, prior=prior)
    assert set(extra) == {"sponsors_studies", "conditions_studies", "facilities_studies"}
    js = db.table_dict["sponsors_studies"].df
    lab = b.history_rows.merge(js[["nct_id", "sponsor_id"]], on="nct_id")
    lab["known"] = pd.to_datetime(lab[t.time_col]) + t.timedelta
    rng = np.random.default_rng(0)
    for i in rng.choice(len(js), 40, replace=False):
        row = js.iloc[i]
        k = lab[(lab.sponsor_id == row.sponsor_id) & (lab.known <= row.date)]
        want = (k[t.target_col].sum() + 5 * prior) / (len(k) + 5)
        assert extra["sponsors_studies"]["sponsor_hist_rate"].iloc[i] == pytest.approx(want)
        assert extra["sponsors_studies"]["sponsor_hist_n"].iloc[i] == pytest.approx(np.log1p(len(k)))


def test_controlled_regimes_differ_only_in_the_quality_column(tmp_path):
    from reltrial import controlled

    h = controlled.build(tmp_path / "h", "hidden", 1.0, n_trials=800, n_sponsors=40)
    v = controlled.build(tmp_path / "v", "visible", 1.0, n_trials=800, n_sponsors=40)
    sh, sv = (pd.read_parquet(p / "db" / "sponsors.parquet") for p in (h, v))
    assert "quality_score" not in sh and "quality_score" in sv
    pd.testing.assert_frame_equal(sh, sv.drop(columns="quality_score"))
    for split in ("train", "val", "test"):
        a, c = (pd.read_parquet(p / "tasks" / "study-outcome" / f"{split}.parquet") for p in (h, v))
        pd.testing.assert_frame_equal(a, c)  # identical labels: only visibility changes
    z = controlled.build(tmp_path / "z", "hidden", 0.0, n_trials=800, n_sponsors=40)
    assert len(pd.read_parquet(z / "tasks" / "study-outcome" / "train.parquet")) > 0


def test_controlled_labels_only_reference_trials_in_the_db(tmp_path):
    """Regression: test seeds after the test timestamp pointed at trials RelBench drops."""
    from reltrial import controlled, data

    b = data.load_bundle("study-outcome", str(controlled.build(tmp_path / "c", "hidden", 1.0,
                                                                n_trials=1500, n_sponsors=40)))
    n = len(b.dataset.get_db().table_dict["studies"].df)
    for s in ("train", "val", "test"):
        assert (b.split(s).df["nct_id"] < n).all(), s
