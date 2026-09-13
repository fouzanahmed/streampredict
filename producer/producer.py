"""Kafka producer: publishes a live synthetic stock-tick stream.

Requires a running Kafka broker (see docker-compose.yml).
Run: python producer/producer.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kafka import KafkaProducer
from tick_generator import generate_tick

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "stock-ticks")
SYMBOLS = os.environ.get("SYMBOLS", "AAPL,MSFT,GOOG").split(",")
TICK_INTERVAL_SECONDS = float(os.environ.get("TICK_INTERVAL_SECONDS", "1.0"))


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    prices = {symbol: 100.0 for symbol in SYMBOLS}
    print(f"Publishing to topic '{TOPIC}' on {BOOTSTRAP_SERVERS} (Ctrl+C to stop)")
    try:
        while True:
            for symbol in SYMBOLS:
                tick = generate_tick(symbol, prices[symbol])
                prices[symbol] = tick.price
                producer.send(TOPIC, tick.to_dict())
                print(tick.to_dict())
            producer.flush()
            time.sleep(TICK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
