import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from train_model import FEATURE_COLS, train


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
