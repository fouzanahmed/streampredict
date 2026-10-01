"""Kafka producer: publishes a live synthetic stock-tick stream.

Requires a running Kafka broker (see docker-compose.yml).
Run: python producer/producer.py

Set LOG_LEVEL=DEBUG to log every published tick.
"""
from __future__ import annotations

import json
import logging
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
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("streampredict.producer")


def configure_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main():
    configure_logging()
    logger.info(
        "starting producer bootstrap_servers=%s topic=%s symbols=%s tick_interval_s=%s",
        BOOTSTRAP_SERVERS, TOPIC, ",".join(SYMBOLS), TICK_INTERVAL_SECONDS,
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    prices = {symbol: 100.0 for symbol in SYMBOLS}
    sent = 0
    logger.info("publishing topic=%s (Ctrl+C to stop)", TOPIC)
    try:
        while True:
            for symbol in SYMBOLS:
                tick = generate_tick(symbol, prices[symbol])
                prices[symbol] = tick.price
                producer.send(TOPIC, tick.to_dict()).add_errback(
                    lambda exc, s=symbol: logger.error("send failed topic=%s symbol=%s error=%r", TOPIC, s, exc)
                )
                sent += 1
                logger.debug(
                    "tick symbol=%s price=%s volume=%s timestamp=%s",
                    tick.symbol, tick.price, tick.volume, tick.timestamp,
                )
            producer.flush()
            time.sleep(TICK_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("interrupted, stopping producer")
    except Exception:
        logger.exception("producer failed ticks_sent=%d", sent)
        raise
    finally:
        producer.close()
        logger.info("producer closed ticks_sent=%d", sent)


if __name__ == "__main__":
    main()
