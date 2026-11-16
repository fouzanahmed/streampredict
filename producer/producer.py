"""Kafka producer: publishes a live synthetic stock-tick stream.

Requires a running Kafka broker (see docker-compose.yml).
Run: python producer/producer.py

Set LOG_LEVEL=DEBUG to log every published tick. Other env vars
(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, SYMBOLS, TICK_INTERVAL_SECONDS) are
parsed and validated in config.py.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from kafka import KafkaProducer
from tick_generator import generate_tick

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("streampredict.producer")


def configure_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main():
    configure_logging()
    try:
        config = load_config()
    except ValueError as exc:
        logger.error("invalid producer config: %s", exc)
        raise SystemExit(2) from exc
    topic = config.topic
    symbols = config.symbols
    logger.info(
        "starting producer bootstrap_servers=%s topic=%s symbols=%s tick_interval_s=%s",
        config.bootstrap_servers, topic, ",".join(symbols), config.tick_interval_seconds,
    )
    producer = KafkaProducer(
        bootstrap_servers=config.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    prices = {symbol: 100.0 for symbol in symbols}
    sent = 0
    logger.info("publishing topic=%s (Ctrl+C to stop)", topic)
    try:
        while True:
            for symbol in symbols:
                tick = generate_tick(symbol, prices[symbol])
                prices[symbol] = tick.price
                producer.send(topic, tick.to_dict()).add_errback(
                    lambda exc, s=symbol: logger.error("send failed topic=%s symbol=%s error=%r", topic, s, exc)
                )
                sent += 1
                logger.debug(
                    "tick symbol=%s price=%s volume=%s timestamp=%s",
                    tick.symbol, tick.price, tick.volume, tick.timestamp,
                )
            producer.flush()
            time.sleep(config.tick_interval_seconds)
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
