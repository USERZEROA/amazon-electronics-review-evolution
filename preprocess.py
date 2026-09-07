"""
Preprocess the Amazon Reviews'23 Electronics dataset.

Outputs:
1. reviews.parquet
   - Full Electronics review dataset
   - Keeps analysis columns but drops full review text
   - Adds review_length_chars

2. reviews_ml_sample.parquet
   - Reproducible sample of positive/negative reviews
   - Keeps review text for machine-learning exploration
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.json as pajson
import pyarrow.parquet as pq


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REVIEW_PATH = RAW_DIR / "Electronics.jsonl.gz"
ANALYSIS_OUTPUT = PROCESSED_DIR / "reviews.parquet"
ML_OUTPUT = PROCESSED_DIR / "reviews_ml_sample.parquet"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Configuration
# ============================================================
RANDOM_SEED = 42
ML_SAMPLE_RATE = 0.005
PROGRESS_EVERY = 1_000_000


# ============================================================
# Explicit schema
# ============================================================

REVIEW_SCHEMA = pa.schema(
    [
        ("rating", pa.float64()),
        ("text", pa.string()),
        ("asin", pa.string()),
        ("parent_asin", pa.string()),
        ("user_id", pa.string()),
        ("timestamp", pa.int64()),
        ("helpful_vote", pa.int64()),
        ("verified_purchase", pa.bool_()),
    ]
)

def preprocess_reviews():

    print(f"Reading: {REVIEW_PATH}")

    read_options = pajson.ReadOptions(
        block_size=64 * 1024 * 1024,
    )

    parse_options = pajson.ParseOptions(
        explicit_schema=REVIEW_SCHEMA,
        unexpected_field_behavior="ignore",
    )

    reader = pajson.open_json(
        REVIEW_PATH,
        read_options=read_options,
        parse_options=parse_options,
    )

    analysis_writer = None
    ml_writer = None

    rng = np.random.default_rng(RANDOM_SEED)

    rows_processed = 0
    next_progress = PROGRESS_EVERY
    ml_rows = 0

    try:
        for batch in reader:

            table = pa.Table.from_batches([batch])

            # ------------------------------------------------
            # Full analysis dataset
            # ------------------------------------------------

            # Treat missing review text as an empty string
            text = pc.fill_null(
                table["text"],
                "",
            )

            review_length_chars = pc.cast(
                pc.utf8_length(text),
                pa.int32(),
            )

            analysis_table = (
                table.drop(["text"])
                .append_column(
                    "review_length_chars",
                    review_length_chars,
                )
            )

            if analysis_writer is None:
                analysis_writer = pq.ParquetWriter(
                    ANALYSIS_OUTPUT,
                    analysis_table.schema,
                    compression="zstd",
                )

            analysis_writer.write_table(
                analysis_table,
                row_group_size=250_000,
            )

            # ------------------------------------------------
            # ML text sample
            # ------------------------------------------------

            ratings = table["rating"].to_numpy(
                zero_copy_only=False
            )

            # ML target:
            # 1–2 stars = negative
            # 4–5 stars = positive
            # 3 stars   = excluded
            non_neutral = (
                (ratings <= 2.0)
                | (ratings >= 4.0)
            )

            random_sample = (
                rng.random(table.num_rows)
                < ML_SAMPLE_RATE
            )

            selected = np.flatnonzero(
                non_neutral & random_sample
            )

            if len(selected) > 0:
                ml_table = table.take(
                    pa.array(selected)
                ).select(
                    [
                        "rating",
                        "text",
                        "parent_asin",
                        "user_id",
                        "timestamp",
                        "verified_purchase",
                    ]
                )

                is_positive = pc.greater_equal(
                    ml_table["rating"],
                    4.0,
                )

                ml_table = ml_table.append_column(
                    "is_positive",
                    is_positive,
                )

                if ml_writer is None:
                    ml_writer = pq.ParquetWriter(
                        ML_OUTPUT,
                        ml_table.schema,
                        compression="zstd",
                    )

                ml_writer.write_table(ml_table)
                ml_rows += ml_table.num_rows

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            rows_processed += table.num_rows

            if rows_processed >= next_progress:
                print(
                    f"Processed {rows_processed:,} reviews "
                    f"| ML sample: {ml_rows:,}"
                )

                while next_progress <= rows_processed:
                    next_progress += PROGRESS_EVERY

    finally:

        if analysis_writer is not None:
            analysis_writer.close()

        if ml_writer is not None:
            ml_writer.close()

    print(f"Total reviews: {rows_processed:,}")
    print(f"ML sample:     {ml_rows:,}")
    print(f"Created: {ANALYSIS_OUTPUT}")
    print(f"Created: {ML_OUTPUT}")


if __name__ == "__main__":
    preprocess_reviews()