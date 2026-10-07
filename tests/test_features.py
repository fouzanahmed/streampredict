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


def test_compute_rolling_features_series_equal_to_window_is_empty():
    # Rolling stats are valid at index window-1, but momentum needs `window`
    # periods back, so a series of exactly `window` points yields nothing.
    prices = pd.Series([10.0, 11.0, 12.0])
    result = compute_rolling_features(prices, window=3)
    assert result.empty
    assert list(result.columns) == ["price", "rolling_mean", "rolling_std", "momentum"]


def test_compute_rolling_features_window_plus_one_points_yields_one_row():
    prices = pd.Series([10.0, 11.0, 12.0, 13.0])
    result = compute_rolling_features(prices, window=3)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["price"] == 13.0
    assert row["rolling_mean"] == pytest.approx(12.0)
    assert row["rolling_std"] == pytest.approx(1.0)
    assert row["momentum"] == pytest.approx((13.0 - 10.0) / 10.0)


def test_compute_rolling_features_with_leading_nan_price():
    # A leading NaN can't be forward-filled, so it delays both the rolling
    # stats (first clean window ends at index 3) and momentum (first clean
    # lookback is index 4 -> index 1). Rows 4, 5, 6 survive.
    prices = pd.Series([float("nan"), 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    result = compute_rolling_features(prices, window=3)
    assert list(result.index) == [4, 5, 6]
    assert result.loc[4, "rolling_mean"] == pytest.approx(13.0)
    assert not result.isna().any().any()


def test_compute_rolling_features_with_mid_series_nan_price():
    # A NaN at index 2 poisons every rolling window containing it (indices
    # 2..4), so no row before index 5 may survive. Whether momentum at index 5
    # survives depends on pct_change's fill behavior (which differs across
    # pandas versions), so only assert the version-independent invariants.
    prices = pd.Series([10.0, 11.0, float("nan"), 13.0, 14.0, 15.0, 16.0])
    result = compute_rolling_features(prices, window=3)
    assert not result.empty
    assert result.index.min() >= 5
    assert 6 in result.index
    assert not result.isna().any().any()
    assert result.loc[6, "rolling_mean"] == pytest.approx(15.0)
    assert result.loc[6, "rolling_std"] == pytest.approx(1.0)


def test_compute_rolling_features_empty_series():
    result = compute_rolling_features(pd.Series([], dtype=float), window=3)
    assert result.empty


def test_add_next_return_label_single_row_is_empty():
    df = pd.DataFrame({"price": [100.0]})
    result = add_next_return_label(df)
    assert result.empty
    assert "next_return" in result.columns


def test_add_next_return_label_drops_rows_touching_nan_price():
    # Row 1's next price is NaN and row 2's own price is NaN, so both are
    # dropped along with the final row; rows 0 and 3 keep valid labels.
    df = pd.DataFrame({"price": [100.0, 110.0, float("nan"), 130.0, 143.0]})
    result = add_next_return_label(df)
    assert list(result.index) == [0, 3]
    assert result.loc[0, "next_return"] == pytest.approx(0.10)
    assert result.loc[3, "next_return"] == pytest.approx(0.10)


def test_add_next_return_label_on_rolling_features_window_plus_two():
    # End-to-end: window+2 points -> 2 feature rows -> 1 labelled row.
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    result = add_next_return_label(compute_rolling_features(prices, window=3))
    assert len(result) == 1
    assert result.iloc[0]["next_return"] == pytest.approx(14.0 / 13.0 - 1)
