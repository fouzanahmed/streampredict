import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from train_model import (
    FEATURE_COLS,
    METADATA_SCHEMA_VERSION,
    TARGET_COL,
    file_sha256,
    metadata_path_for,
    save_model,
    train,
)

CONSUMER_JOB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "streaming", "consumer_job.py"
)


@pytest.fixture
def synthetic_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    rolling_mean = rng.normal(100, 5, n)
    rolling_std = rng.uniform(0.1, 2.0, n)
    next_return = rng.normal(0, 0.01, n)
    df = pd.DataFrame({
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "momentum": rng.normal(0, 0.02, n),
        "next_return": next_return,
    })
    path = tmp_path / "training_data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_train_returns_fitted_model_and_metrics(synthetic_csv):
    model, metrics = train(data_path=synthetic_csv)
    assert hasattr(model, "predict")
    assert set(metrics.keys()) == {"mae", "naive_mae", "n_train", "n_test"}
    assert metrics["n_train"] + metrics["n_test"] == 200


def test_trained_model_predicts_expected_shape(synthetic_csv):
    model, _ = train(data_path=synthetic_csv)
    X = pd.DataFrame({"rolling_mean": [100.0, 101.0], "rolling_std": [1.0, 1.2]})
    assert list(X.columns) == FEATURE_COLS
    preds = model.predict(X)
    assert preds.shape == (2,)


def test_save_model_writes_metadata_alongside_artifact(synthetic_csv, tmp_path):
    model, metrics = train(data_path=synthetic_csv)
    model_path = str(tmp_path / "artifacts" / "model.pkl")
    meta_path = save_model(model, metrics, model_path=model_path)

    assert meta_path == metadata_path_for(model_path)
    assert meta_path.endswith("model.meta.json")
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["schema_version"] == METADATA_SCHEMA_VERSION
    assert metadata["feature_cols"] == FEATURE_COLS
    assert metadata["target_col"] == TARGET_COL
    assert metadata["model_type"] == "LinearRegression"
    assert metadata["metrics"]["mae"] == pytest.approx(metrics["mae"])
    assert metadata["metrics"]["n_train"] + metadata["metrics"]["n_test"] == 200
    assert metadata["model_sha256"] == file_sha256(model_path)
    assert "trained_at" in metadata


def _consumer_constant(name):
    # consumer_job.py imports pyspark at module level, which CI doesn't
    # install, so read its constants from the source instead of importing it.
    with open(CONSUMER_JOB_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in consumer_job.py")


def test_consumer_expects_the_features_and_schema_the_trainer_writes():
    assert _consumer_constant("EXPECTED_FEATURE_COLS") == FEATURE_COLS
    assert METADATA_SCHEMA_VERSION in _consumer_constant("SUPPORTED_METADATA_SCHEMA_VERSIONS")
