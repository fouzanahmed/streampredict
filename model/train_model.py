"""Trains a baseline next-tick-return regressor on the synthetic dataset
produced by generate_training_data.py, and saves it for use by the
streaming inference job.

Run: python model/generate_training_data.py && python model/train_model.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Only rolling_mean/rolling_std are used by the deployed model: these are
# the two features the Spark Structured Streaming job can compute directly
# from a tumbling window aggregation (avg/stddev). `momentum` is available
# in the offline dataset (see model/features.py) for further analysis, but
# a streaming tumbling window can't reproduce a trailing-window pct_change
# without stateful lag joins, which is out of scope for this MVP -- so the
# model is deliberately trained on only the features the streaming path can
# actually deliver in real time, to keep training and inference consistent.
FEATURE_COLS = ["rolling_mean", "rolling_std"]
TARGET_COL = "next_return"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "training_data.csv")
MODEL_PATH = os.path.join(HERE, "artifacts", "model.pkl")

# Bump when the metadata layout or the model's input contract changes in a
# way streaming/consumer_job.py must know about; the consumer refuses to
# load a model whose metadata schema version it doesn't recognise.
METADATA_SCHEMA_VERSION = 1


def metadata_path_for(model_path: str) -> str:
    """Metadata lives next to the model: model.pkl -> model.meta.json."""
    return os.path.splitext(model_path)[0] + ".meta.json"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def train(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # Naive baseline for comparison: always predict zero return.
    naive_mae = mean_absolute_error(y_test, [0.0] * len(y_test))

    return model, {"mae": mae, "naive_mae": naive_mae, "n_train": len(X_train), "n_test": len(X_test)}


def build_metadata(model, metrics: dict, model_sha256: str) -> dict:
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "sklearn_version": sklearn.__version__,
        "feature_cols": list(FEATURE_COLS),
        "target_col": TARGET_COL,
        "metrics": {
            "mae": float(metrics["mae"]),
            "naive_mae": float(metrics["naive_mae"]),
            "n_train": int(metrics["n_train"]),
            "n_test": int(metrics["n_test"]),
        },
        "model_sha256": model_sha256,
    }


def save_model(model, metrics: dict, model_path: str = MODEL_PATH) -> str:
    """Pickle the model and write its metadata file alongside it.

    Returns the metadata path. The metadata records a hash of the pickle so
    the consumer can detect a model file that doesn't match its metadata.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    meta_path = metadata_path_for(model_path)
    metadata = build_metadata(model, metrics, file_sha256(model_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return meta_path


if __name__ == "__main__":
    model, metrics = train()
    meta_path = save_model(model, metrics)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved model metadata to {meta_path}")
    print(f"Observed on held-out test split: {metrics}")
