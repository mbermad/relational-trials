import json

import numpy as np
import pytest

from reltrial.evaluate import RunRecord, compute_metrics, load_runs, mean_ci, paired_bootstrap


def test_metrics_classification():
    m = compute_metrics("binary_classification", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m == {"roc_auc": 1.0}


def test_metrics_regression_nmae():
    m = compute_metrics("regression", [0, 2, 4], [1, 2, 3], nmae_std=2.0)
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["nmae"] == pytest.approx(1 / 3)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics("regression", [1, 2], [1])


def test_mean_ci():
    mean, half, n = mean_ci([1.0, 2.0, 3.0])
    assert (mean, n) == (2.0, 3)
    assert half == pytest.approx(2.484, abs=1e-3)  # t(0.975, 2) * 1 / sqrt(3)
    assert mean_ci([5.0])[1] != mean_ci([5.0])[1]  # NaN half-width for one seed


def test_paired_bootstrap_prefers_better_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.5, 400)
    bad = rng.normal(0, 1, 400)
    r = paired_bootstrap(y, good, bad, "roc_auc", n_boot=300)
    assert r["p_a_better"] > 0.99 and r["ci_low"] > 0


def test_run_record_roundtrip(tmp_path):
    rec = RunRecord(task="t", model="m", seed=0, task_type="regression",
                    val={"mae": 1.0}, test={"mae": 2.0}, train_seconds=3.0)
    path = rec.save(tmp_path)
    assert json.loads(path.read_text())["test"]["mae"] == 2.0
    assert load_runs(tmp_path)[0]["model"] == "m"
