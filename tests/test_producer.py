import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "producer"))

from tick_generator import generate_stream, generate_tick, next_price


def test_next_price_stays_positive_under_extreme_volatility():
    rng = random.Random(0)
    price = 0.02
    for _ in range(1000):
        price = next_price(price, volatility=0.5, rng=rng)
        assert price > 0


def test_generate_tick_schema():
    tick = generate_tick("AAPL", 100.0, rng=random.Random(1))
    d = tick.to_dict()
    assert set(d.keys()) == {"symbol", "price", "volume", "timestamp"}
    assert d["symbol"] == "AAPL"
    assert d["price"] > 0
    assert 1 <= d["volume"] <= 500


def test_generate_stream_length_and_order():
    ticks = generate_stream(["AAPL", "MSFT"], n_per_symbol=10, seed=42)
    assert len(ticks) == 20
    aapl_ticks = [t for t in ticks if t.symbol == "AAPL"]
    assert len(aapl_ticks) == 10
    # timestamps should be non-decreasing within a symbol's sequence
    timestamps = [t.timestamp for t in aapl_ticks]
    assert timestamps == sorted(timestamps)


def test_generate_stream_is_deterministic_with_seed():
    a = generate_stream(["AAPL"], n_per_symbol=50, seed=7)
    b = generate_stream(["AAPL"], n_per_symbol=50, seed=7)
    assert [t.price for t in a] == [t.price for t in b]
