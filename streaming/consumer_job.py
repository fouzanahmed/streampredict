"""PySpark Structured Streaming job: consumes the live tick stream from
Kafka, computes windowed rolling features per symbol, applies the trained
model for real-time inference, and writes predictions to console + Parquet.

Requires a running Kafka broker and a trained model (see
model/train_model.py). Run with spark-submit, e.g.:

  spark-submit \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
    streaming/consumer_job.py
"""
from __future__ import annotations

import os
import pickle

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, from_json, pandas_udf, stddev, window
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "stock-ticks")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "streaming/output")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "streaming/checkpoint")
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "artifacts", "model.pkl")
)

TICK_SCHEMA = StructType([
    StructField("symbol", StringType()),
    StructField("price", DoubleType()),
    StructField("volume", LongType()),
    StructField("timestamp", DoubleType()),
])


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_predict_udf():
    model = load_model()

    @pandas_udf(DoubleType())
    def predict(rolling_mean: pd.Series, rolling_std: pd.Series) -> pd.Series:
        X = pd.DataFrame({
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
        }).fillna(0.0)
        return pd.Series(model.predict(X))

    return predict


def main():
    spark = SparkSession.builder.appName("StreamPredict").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    ticks = raw.select(from_json(col("value").cast("string"), TICK_SCHEMA).alias("tick")).select("tick.*")
    ticks = ticks.withColumn("event_time", (col("timestamp")).cast("timestamp"))

    # Windowed rolling stats per symbol (5-minute tumbling window as a
    # streaming-friendly stand-in for the row-based rolling window used
    # offline in model/features.py -- see train_model.py for why the model
    # only uses these two features).
    windowed = (
        ticks.groupBy(window(col("event_time"), "5 minutes"), col("symbol"))
        .agg(
            avg("price").alias("rolling_mean"),
            stddev("price").alias("rolling_std"),
        )
    )

    predict = build_predict_udf()
    scored = windowed.withColumn(
        "predicted_next_return",
        predict(col("rolling_mean"), col("rolling_std")),
    )

    query = (
        scored.writeStream.outputMode("update")
        .format("console")
        .option("truncate", False)
        .start()
    )

    parquet_query = (
        scored.writeStream.outputMode("append")
        .format("parquet")
        .option("path", OUTPUT_PATH)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .start()
    )

    query.awaitTermination()
    parquet_query.awaitTermination()


if __name__ == "__main__":
    main()
