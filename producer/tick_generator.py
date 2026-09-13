"""Pure, testable synthetic stock-tick generation (random-walk prices).

Used by both the live Kafka producer and the offline training-data
generator, so the two stay consistent with each other.
"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass


@dataclass
class Tick:
    symbol: str
    price: float
    volume: int
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


def next_price(prev_price: float, volatility: float = 0.002, rng: random.Random | None = None) -> float:
    """One random-walk step. Volatility is a per-tick fractional std-dev."""
    rng = rng or random
    pct_change = rng.gauss(0, volatility)
    new_price = prev_price * (1 + pct_change)
    return max(new_price, 0.01)


def generate_tick(symbol: str, prev_price: float, rng: random.Random | None = None) -> Tick:
    rng = rng or random
    price = next_price(prev_price, rng=rng)
    volume = rng.randint(1, 500)
    return Tick(symbol=symbol, price=round(price, 4), volume=volume, timestamp=time.time())


def generate_stream(symbols: list[str], n_per_symbol: int, start_price: float = 100.0,
                     seed: int | None = None) -> list[Tick]:
    """Generate n_per_symbol sequential ticks for each symbol (in-order per symbol)."""
    rng = random.Random(seed)
    ticks: list[Tick] = []
    for symbol in symbols:
        price = start_price
        for _ in range(n_per_symbol):
            t = generate_tick(symbol, price, rng=rng)
            ticks.append(t)
            price = t.price
    return ticks
