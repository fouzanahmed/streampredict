import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from features import add_next_return_label, compute_rolling_features


def test_compute_rolling_features_known_values():
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    result = compute_rolling_features(prices, window=3)
    # momentum needs `window` periods back (valid from index 3), which is
    # more restrictive than rolling mean/std (valid from index 2), so only
    # indices 3,4,5 survive the combined dropna -> 3 rows.
    assert len(result) == 3
    # Row for price=13.0 (index 3): rolling mean of [11,12,13] = 12.0
    row = result[result["price"] == 13.0].iloc[0]
    assert row["rolling_mean"] == 12.0
    assert row["momentum"] == pytest.approx((13.0 - 10.0) / 10.0)


def test_add_next_return_label_shifts_correctly():
    df = pd.DataFrame({"price": [100.0, 110.0, 121.0]})
    result = add_next_return_label(df)
    # Last row dropped (no next price)
    assert len(result) == 2
    assert result.iloc[0]["next_return"] == pytest.approx(0.10)
    assert result.iloc[1]["next_return"] == pytest.approx(0.10)


def test_rolling_features_drops_nan_rows():
    prices = pd.Series([1.0, 2.0])
    result = compute_rolling_features(prices, window=5)
    assert len(result) == 0
