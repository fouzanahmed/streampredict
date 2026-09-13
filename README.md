# StreamPredict

A real-time streaming ML pipeline: a Kafka producer simulates a live stock-tick stream, a PySpark Structured Streaming job aggregates it into rolling per-symbol windows and scores each window with a trained regression model, writing predictions to console and Parquet.

## Architecture

```
producer/producer.py  -->  Kafka topic "stock-ticks"  -->  streaming/consumer_job.py (PySpark Structured Streaming)
                                                                |
                                                                +--> console output
                                                                +--> Parquet files (streaming/output/)
```

- **`producer/tick_generator.py`** — pure, seedable random-walk tick generation, shared by the live producer and the offline training-data generator so both produce data the same way.
- **`producer/producer.py`** — publishes a continuous synthetic tick stream to Kafka.
- **`model/generate_training_data.py`** + **`model/features.py`** — builds a synthetic historical dataset with rolling-window features (mean, std, momentum) and a next-tick-return label.
- **`model/train_model.py`** — trains a scikit-learn linear regression to predict next-tick return from `rolling_mean`/`rolling_std`. It deliberately does **not** use `momentum` even though the offline dataset has it: a Spark Structured Streaming tumbling window can compute `avg`/`stddev` directly, but reproducing a trailing-window `pct_change` in a streaming context needs stateful lag joins, which is out of scope for this MVP — so the model is trained only on the two features the streaming path can actually deliver in real time.
- **`streaming/consumer_job.py`** — the Spark Structured Streaming job: reads from Kafka, computes 5-minute tumbling-window `rolling_mean`/`rolling_std` per symbol, applies the trained model via a pandas UDF, writes results.

## What's actually verified vs. not

This was built and tested in one pass; here's the honest status:

- **Verified, actually run:** `producer/tick_generator.py`, `model/features.py`, `model/train_model.py`, and all 9 pytest tests — installed a clean venv, ran the full offline pipeline end to end (`generate_training_data.py` → `train_model.py`), and ran `ruff check` (0 issues after one auto-fix pass).
- **Observed result** (from an actual run, not invented): on a held-out test split of synthetic random-walk data, the trained model's MAE (0.00159) was essentially identical to a naive "always predict zero return" baseline (0.00159). That's the expected, correct outcome for a pure random walk with no injected signal — it demonstrates the training/inference pipeline works end to end, **not** that the model has predictive edge. A real deployment would need real market data and feature engineering to do better than the naive baseline.
- **Not verified (syntax-checked only):** `producer/producer.py` and `streaming/consumer_job.py` — these need a running Kafka broker and a Spark runtime respectively, which weren't exercised end-to-end in this environment. Both pass a Python compile check (no syntax errors), but haven't been run against a live Kafka topic.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 1. Generate synthetic training data and train the model
python model/generate_training_data.py
python model/train_model.py

# 2. Start Kafka locally
docker compose up -d

# 3. In one terminal: start producing ticks
python producer/producer.py

# 4. In another terminal: run the Spark Structured Streaming job
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 streaming/consumer_job.py
```

## Technologies

Kafka, PySpark Structured Streaming, pandas, scikit-learn, Docker.

## Tests

```bash
pytest tests/
```

Covers the tick generator, feature engineering, and model training/inference — all pure-Python logic that doesn't require a live Kafka or Spark cluster. `streaming/consumer_job.py` and `producer/producer.py` aren't covered by these tests since they need live infrastructure; CI only lints and compile-checks them.
