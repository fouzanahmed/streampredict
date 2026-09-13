"""Generates a synthetic historical tick dataset (using the same random-walk
generator as the live producer) and writes rolling features + labels to CSV,
so the whole pipeline is reproducible without any external data dependency.

Run: python model/generate_training_data.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "producer"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from features import add_next_return_label, compute_rolling_features
from tick_generator import generate_stream

SYMBOLS = ["AAPL", "MSFT", "GOOG"]
N_PER_SYMBOL = 2000
WINDOW = 5
SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.csv")


def build_dataset() -> pd.DataFrame:
    ticks = generate_stream(SYMBOLS, N_PER_SYMBOL, seed=SEED)
    raw = pd.DataFrame([t.to_dict() for t in ticks])

    frames = []
    for symbol, group in raw.groupby("symbol"):
        prices = group["price"].reset_index(drop=True)
        feats = compute_rolling_features(prices, window=WINDOW)
        feats = add_next_return_label(feats)
        feats["symbol"] = symbol
        frames.append(feats)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    dataset = build_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(dataset)} rows to {OUTPUT_PATH}")
