"""Environment config for the Kafka producer.

Kept free of kafka imports so the parsing and validation can be unit-tested
without a broker or the kafka-python package.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping

DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC = "stock-ticks"
DEFAULT_SYMBOLS = "AAPL,MSFT,GOOG"
DEFAULT_TICK_INTERVAL_SECONDS = "1.0"


@dataclass(frozen=True)
class ProducerConfig:
    bootstrap_servers: str
    topic: str
    symbols: tuple[str, ...]
    tick_interval_seconds: float


def parse_symbols(raw: str) -> tuple[str, ...]:
    """Split a comma-separated symbol list, stripping whitespace and dropping blanks.

    Duplicates are removed, keeping the first occurrence. Raises ValueError if
    no symbols are left.
    """
    symbols: list[str] = []
    for part in raw.split(","):
        symbol = part.strip()
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        raise ValueError(
            f"SYMBOLS must contain at least one non-empty symbol, got {raw!r}"
        )
    return tuple(symbols)


def parse_tick_interval(raw: str) -> float:
    """Parse TICK_INTERVAL_SECONDS as a finite, positive number of seconds."""
    try:
        interval = float(raw.strip())
    except ValueError:
        raise ValueError(
            f"TICK_INTERVAL_SECONDS must be a number, got {raw!r}"
        ) from None
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError(
            f"TICK_INTERVAL_SECONDS must be a finite number greater than 0, got {raw!r}"
        )
    return interval


def load_config(env: Mapping[str, str] | None = None) -> ProducerConfig:
    """Build a ProducerConfig from env (defaults to os.environ)."""
    if env is None:
        env = os.environ
    return ProducerConfig(
        bootstrap_servers=env.get("KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS),
        topic=env.get("KAFKA_TOPIC", DEFAULT_TOPIC),
        symbols=parse_symbols(env.get("SYMBOLS", DEFAULT_SYMBOLS)),
        tick_interval_seconds=parse_tick_interval(
            env.get("TICK_INTERVAL_SECONDS", DEFAULT_TICK_INTERVAL_SECONDS)
        ),
    )
