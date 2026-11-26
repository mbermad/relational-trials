"""End-to-end smoke tests on the synthetic rel-trial-shaped dataset."""

import importlib.util

import pytest

import synthetic
from reltrial import data, lgbm, tables
from reltrial.evaluate import load_runs


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    return str(synthetic.build(tmp_path_factory.mktemp("syn") / "rel-trial-synthetic"))


@pytest.fixture(scope="module")
def bundles(dataset_dir):
    out = {"study-outcome": data.load_bundle("study-outcome", dataset_dir)}
    out["study-adverse"] = data.load_bundle("study-adverse", dataset_dir, dataset=out["study-outcome"].dataset)
    return out


def test_test_labels_are_available_offline(bundles):
    b = bundles["study-outcome"]
    assert b.task.target_col in b.test.df.columns
    assert len(b.test.df) > 0


def test_lgbm_runs_and_relational_features_help(bundles, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    b = bundles["study-outcome"]
    table = lgbm.run(b, "table", seeds=[0], n_trials=2, out_dir="runs", save_preds="preds")
    agg = lgbm.run(b, "agg", seeds=[0], n_trials=2, out_dir="runs", save_preds="preds")
    # The synthetic labels are driven by sponsor history, which only 'agg' can see.
    assert agg[0].test["roc_auc"] > table[0].test["roc_auc"]
    assert agg[0].test["roc_auc"] > 0.65
    assert len(load_runs("runs")) == 2
    tables.build("runs", "tables")
    assert (tmp_path / "tables" / "main_rows.tex").exists()


def test_regression_reports_mae_and_nmae(bundles, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = lgbm.run(bundles["study-adverse"], "table", seeds=[0], n_trials=1, out_dir="runs",
                   save_preds=None)[0]
    assert set(rec.test) == {"mae", "nmae"}


@pytest.mark.skipif(any(importlib.util.find_spec(m) is None
                        for m in ("torch_geometric", "torch_frame", "pyg_lib")),
                    reason="GNN dependencies (torch_geometric, torch_frame, pyg-lib) not installed")
def test_rdl_smoke(bundles, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from reltrial import rdl

    b = bundles["study-outcome"]
    graph, stats = rdl.build_graph(b, cache_dir=str(tmp_path / "cache"), text="hash")
    recs = rdl.run(b, graph, stats, seeds=[0], epochs=2, channels=32, num_neighbors=16,
                   batch_size=128, out_dir="runs", save_preds=None, save_embeddings="emb")
    assert 0.0 <= recs[0].test["roc_auc"] <= 1.0
    assert (tmp_path / "emb" / "study-outcome" / "rdl_sage" / "test.npy").exists()


def test_nohist_ablation_drops_only_label_history(bundles):
    b = bundles["study-outcome"]
    X_agg, _ = lgbm.build_features(b, "agg")
    X_nohist, _ = lgbm.build_features(b, "agg_nohist")
    hist_cols = {c for c in X_agg["train"].columns if "_hist_" in c}
    assert hist_cols, "the full aggregate set should contain label-history features"
    assert not any("_hist_" in c for c in X_nohist["train"].columns)
    assert set(X_agg["train"].columns) - set(X_nohist["train"].columns) == hist_cols


def test_label_history_explains_the_synthetic_signal(bundles, tmp_path, monkeypatch):
    # In the synthetic data, outcomes are driven by the sponsor's latent quality, which only
    # the history of its earlier trials reveals: removing history should hurt.
    monkeypatch.chdir(tmp_path)
    b = bundles["study-outcome"]
    full = lgbm.run(b, "agg", seeds=[0], n_trials=2, out_dir="runs", save_preds=None)[0]
    nohist = lgbm.run(b, "agg_nohist", seeds=[0], n_trials=2, out_dir="runs", save_preds=None)[0]
    assert full.test["roc_auc"] > nohist.test["roc_auc"] + 0.03


@pytest.mark.skipif(any(importlib.util.find_spec(m) is None
                        for m in ("torch_geometric", "torch_frame", "pyg_lib")),
                    reason="GNN dependencies (torch_geometric, torch_frame, pyg-lib) not installed")
def test_rdl_with_label_history_smoke(bundles, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from reltrial import rdl

    b = bundles["study-outcome"]
    hist = rdl.history_features(b)
    assert hist["train"].shape[0] == len(b.train.df)
    assert hist["test"].shape[0] == len(b.test.df)
    graph, stats = rdl.build_graph(b, cache_dir=str(tmp_path / "cache"), text="hash")
    recs = rdl.run(b, graph, stats, seeds=[0], epochs=2, channels=32, num_neighbors=16,
                   batch_size=128, out_dir="runs", save_preds=None, hist=hist,
                   model_name="rdl_sage_hist")
    assert recs[0].params["label_history_features"] == hist["train"].shape[1]
    assert 0.0 <= recs[0].test["roc_auc"] <= 1.0
