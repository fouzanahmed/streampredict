"""Rolling feature engineering, shared by offline training and the streaming
inference job so both compute features the same way.
"""
from __future__ import annotations

import pandas as pd


def compute_rolling_features(prices: pd.Series, window: int = 5) -> pd.DataFrame:
    """Given a per-symbol, time-ordered price series, compute:
      - rolling_mean: mean price over the trailing window
      - rolling_std: price volatility over the trailing window
      - momentum: percent change over the trailing window (kept for offline
        analysis; NOT used by the deployed model -- see note in train_model.py
        on why the streaming job only reproduces rolling_mean/rolling_std)
    Rows before `window` observations are available are dropped (NaN rolling stats).
    """
    df = pd.DataFrame({"price": prices})
    df["rolling_mean"] = df["price"].rolling(window).mean()
    df["rolling_std"] = df["price"].rolling(window).std()
    df["momentum"] = df["price"].pct_change(periods=window)
    return df.dropna()


def add_next_return_label(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Adds `next_return`: the percent change from this row's price to the
    NEXT row's price. The last row has no next price and is dropped."""
    df = df.copy()
    df["next_return"] = df[price_col].shift(-1) / df[price_col] - 1
    return df.dropna()
