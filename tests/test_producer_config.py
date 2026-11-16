import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "producer"))

from config import ProducerConfig, load_config, parse_symbols, parse_tick_interval


def test_parse_symbols_strips_whitespace_and_drops_empty_entries():
    assert parse_symbols("AAPL, MSFT,") == ("AAPL", "MSFT")
    assert parse_symbols("  AAPL ,, ,GOOG  ") == ("AAPL", "GOOG")


def test_parse_symbols_drops_duplicates_keeping_first_order():
    assert parse_symbols("MSFT,AAPL, MSFT") == ("MSFT", "AAPL")


@pytest.mark.parametrize("raw", ["", " ", ",", " , ,, "])
def test_parse_symbols_rejects_empty_list(raw):
    with pytest.raises(ValueError, match="SYMBOLS"):
        parse_symbols(raw)


@pytest.mark.parametrize("raw, expected", [("1.0", 1.0), (" 0.25 ", 0.25), ("3", 3.0)])
def test_parse_tick_interval_accepts_positive_numbers(raw, expected):
    assert parse_tick_interval(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "-0.5", "nan", "inf", "-inf"])
def test_parse_tick_interval_rejects_non_positive_or_non_finite(raw):
    with pytest.raises(ValueError, match="greater than 0"):
        parse_tick_interval(raw)


@pytest.mark.parametrize("raw", ["", "abc", "1s", "1,5"])
def test_parse_tick_interval_rejects_non_numeric(raw):
    with pytest.raises(ValueError, match="must be a number"):
        parse_tick_interval(raw)


def test_load_config_defaults_when_env_is_empty():
    assert load_config({}) == ProducerConfig(
        bootstrap_servers="localhost:9092",
        topic="stock-ticks",
        symbols=("AAPL", "MSFT", "GOOG"),
        tick_interval_seconds=1.0,
    )


def test_load_config_reads_and_normalises_env_values():
    cfg = load_config({
        "KAFKA_BOOTSTRAP_SERVERS": "kafka:29092",
        "KAFKA_TOPIC": "ticks",
        "SYMBOLS": "AAPL, MSFT,",
        "TICK_INTERVAL_SECONDS": "0.5",
    })
    assert cfg.bootstrap_servers == "kafka:29092"
    assert cfg.topic == "ticks"
    assert cfg.symbols == ("AAPL", "MSFT")
    assert cfg.tick_interval_seconds == 0.5


def test_load_config_surfaces_invalid_interval():
    with pytest.raises(ValueError, match="TICK_INTERVAL_SECONDS"):
        load_config({"TICK_INTERVAL_SECONDS": "-2"})


def test_load_config_uses_os_environ_by_default(monkeypatch):
    monkeypatch.setenv("SYMBOLS", "TSLA")
    monkeypatch.setenv("TICK_INTERVAL_SECONDS", "2")
    cfg = load_config()
    assert cfg.symbols == ("TSLA",)
    assert cfg.tick_interval_seconds == 2.0
