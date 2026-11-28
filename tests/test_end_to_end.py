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


def test_relgnn_atomic_routes_skip_junction_tables():
    from reltrial.relgnn.atomic_routes import get_atomic_routes

    # Edge types as RelBench builds them for studies <- sponsors_studies -> sponsors.
    edges = [("sponsors_studies", "f2p_nct_id", "studies"),
             ("studies", "rev_f2p_nct_id", "sponsors_studies"),
             ("sponsors_studies", "f2p_sponsor_id", "sponsors"),
             ("sponsors", "rev_f2p_sponsor_id", "sponsors_studies"),
             ("designs", "f2p_nct_id", "studies"),
             ("studies", "rev_f2p_nct_id", "designs")]
    routes = get_atomic_routes(edges)
    # A composite route lets a sponsor's message reach the trial in one layer.
    assert ("dim-fact-dim", "sponsors_studies", "f2p_nct_id", "studies",
            "sponsors", "rev_f2p_sponsor_id", "sponsors_studies") in routes
    # A table with a single foreign key keeps an ordinary (dim-dim) edge.
    assert ("dim-dim", "designs", "f2p_nct_id", "studies") in routes


@pytest.mark.skipif(any(importlib.util.find_spec(m) is None
                        for m in ("torch_geometric", "torch_frame", "pyg_lib")),
                    reason="GNN dependencies (torch_geometric, torch_frame, pyg-lib) not installed")
def test_relgnn_and_tuning_smoke(bundles, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from reltrial import rdl

    b = bundles["study-outcome"]
    graph, stats = rdl.build_graph(b, cache_dir=str(tmp_path / "cache"), text="hash")
    cfg = rdl.relgnn_config("rel-trial", "study-outcome")
    cfg.update(channels=32, num_neighbors=16, batch_size=128)
    recs = rdl.run(b, graph, stats, seeds=[0], epochs=2, out_dir="runs", save_preds=None,
                   arch="relgnn", model_name="relgnn", dropout=0.2, weight_decay=1e-4, **cfg)
    assert 0.0 <= recs[0].test["roc_auc"] <= 1.0
    best = rdl.tune(b, graph, stats, n_trials=2, epochs=2, batch_size=128, patience=1,
                    base={"lr": 0.005, "channels": 32, "dropout": 0.0, "weight_decay": 0.0,
                          "num_neighbors": 16})
    assert set(best) == set(rdl.TUNE_SPACE)


def test_f1_shaped_dataset_own_history_helps(tmp_path_factory, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = synthetic.build_f1(tmp_path_factory.mktemp("f1") / "rel-f1-synthetic")
    b = data.load_bundle("driver-dnf", str(root))
    full = lgbm.run(b, "agg", seeds=[0], n_trials=2, out_dir="runs", save_preds="preds")[0]
    nohist = lgbm.run(b, "agg_nohist", seeds=[0], n_trials=2, out_dir="runs", save_preds="preds")[0]
    assert full.test["roc_auc"] > nohist.test["roc_auc"] + 0.03
    X, _ = lgbm.build_features(b, "agg")
    assert {"own_hist_target_mean", "constructor_prior_trials_mean"} <= set(X["train"].columns)


def test_max_train_subsamples_fitting_rows_but_keeps_history(tmp_path_factory):
    root = synthetic.build_f1(tmp_path_factory.mktemp("f1sub") / "rel-f1-synthetic")
    full = data.load_bundle("driver-dnf", str(root))
    n = len(full.train.df) // 4
    sub = data.load_bundle("driver-dnf", str(root), dataset=full.dataset, max_train=n)
    assert len(sub.train.df) <= n + full.train.df[full.task.time_col].value_counts().max()
    assert len(sub.train.df) < len(full.train.df) and len(sub.train_full.df) == len(full.train.df)
    assert sub.train.df[sub.task.time_col].min() > full.train.df[full.task.time_col].min()
    assert len(sub.history_rows) == len(full.history_rows)
    Xs, _ = lgbm.build_features(sub, "agg")
    Xf, _ = lgbm.build_features(full, "agg")
    # The latest training rows see the same history either way.
    tail = Xf["train"].iloc[-len(Xs["train"]):]["own_hist_n_sum"].to_numpy()
    assert (Xs["train"]["own_hist_n_sum"].to_numpy() == tail).all()
