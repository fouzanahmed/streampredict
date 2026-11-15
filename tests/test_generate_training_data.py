import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from generate_training_data import N_PER_SYMBOL, SYMBOLS, WINDOW, build_dataset

EXPECTED_COLUMNS = {"price", "rolling_mean", "rolling_std", "momentum", "next_return", "symbol"}


@pytest.fixture(scope="module")
def dataset():
    return build_dataset()


def test_build_dataset_has_expected_columns(dataset):
    assert set(dataset.columns) == EXPECTED_COLUMNS


def test_build_dataset_contains_every_symbol(dataset):
    assert set(dataset["symbol"].unique()) == set(SYMBOLS)


def test_build_dataset_row_count(dataset):
    # Per symbol: the first WINDOW rows are dropped (momentum's pct_change
    # lookback is WINDOW, one longer than the rolling warmup of WINDOW-1),
    # then the last row is dropped because it has no next price.
    per_symbol = N_PER_SYMBOL - WINDOW - 1
    assert len(dataset) == len(SYMBOLS) * per_symbol
    counts = dataset["symbol"].value_counts()
    for symbol in SYMBOLS:
        assert counts[symbol] == per_symbol


def test_build_dataset_has_no_nans(dataset):
    assert not dataset.isna().any().any()


def test_build_dataset_index_is_contiguous(dataset):
    # concat(ignore_index=True) should yield a clean 0..n-1 RangeIndex.
    assert list(dataset.index) == list(range(len(dataset)))


def test_build_dataset_is_deterministic_with_seed(dataset):
    second = build_dataset()
    pd.testing.assert_frame_equal(dataset, second)
