"""Trains a baseline next-tick-return regressor on the synthetic dataset
produced by generate_training_data.py, and saves it for use by the
streaming inference job.

Run: python model/generate_training_data.py && python model/train_model.py
"""
from __future__ import annotations

import os
import pickle

import pandas as pd
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
TEST_SIZE = 0.2
# Minimum rows needed for a meaningful 80/20 split: at least a couple of
# held-out rows and enough training rows to fit the two-feature regressor.
MIN_ROWS = 10

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "training_data.csv")
MODEL_PATH = os.path.join(HERE, "artifacts", "model.pkl")


def validate_training_frame(df: pd.DataFrame, source: str = "training data") -> None:
    """Raise a descriptive ValueError if `df` can't be used to train the model."""
    required = FEATURE_COLS + [TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s) {missing}; "
            f"expected {required}, found {list(df.columns)}. "
            "Regenerate it with `python model/generate_training_data.py`."
        )
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"{source} has {len(df)} row(s); at least {MIN_ROWS} are needed "
            f"for a train/test split with test_size={TEST_SIZE}."
        )


def train(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path)
    validate_training_frame(df, source=f"Training data at {data_path!r}")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    # Naive baseline for comparison: always predict zero return.
    naive_mae = mean_absolute_error(y_test, [0.0] * len(y_test))

    return model, {"mae": mae, "naive_mae": naive_mae, "n_train": len(X_train), "n_test": len(X_test)}


if __name__ == "__main__":
    model, metrics = train()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Observed on held-out test split: {metrics}")
